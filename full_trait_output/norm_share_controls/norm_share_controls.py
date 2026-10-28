#!/usr/bin/env python3
"""
Sanity checks for the "share of squared norm in persona space" numbers of
Section sec:directions / fig:direction-story (Llama-3.1-8B, layer 16, last prompt token):
refusal direction 0.407, jailbreak direction (dmu, all data) 0.482, vs 229/4096 = 0.056.

The 0.056 baseline assumes an isotropic direction. Activations are not isotropic, and the
trait vectors are themselves diff-of-means of activations, so any diff-of-means could land in
the persona span just because both sit in the high-variance directions of the residual stream.
The controls below separate that from refusal/jailbreak-specific alignment.

  share_S(d) = ||Q_S^T d||^2 / ||d||^2,  Q_S an orthonormal basis of subspace S.

  A. Subspaces, targets fixed (refusal, dmu):
     persona span (229 traits, QR, as in the paper) | random 229-dim (200 draws) |
     top-229 PCs of Alpaca activations (centered) | span of 229 random Alpaca activations (200 draws)
  B. Directions, subspace fixed (persona span):
     refusal, dmu | Alpaca mean activation | random Gaussian (200) |
     shuffled-label refusal (harmful vs Alpaca, labels permuted, 200) |
     shuffled-label dmu (HB jailbroken vs refused, labels permuted, 200)
  C. How much of Alpaca's centered activation variance each 229-dim subspace holds.

Alpaca = the 797 plain Alpaca instructions of the v2 role-play set, harmful = its 897 plain
JailbreakBench/AdvBench/StrongREJECT requests, both read at L16 last prompt token with the
official template (roleplay_check/cluster_v2.py). The rebuilt harmful-minus-Alpaca direction is
compared with Arditi's saved direction as a check that it is a refusal direction.
No model is run; saved activations only. Writes results.json and tables.md next to this file.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "full_trait_tools"))
from run_all_traits_sweep_v2 import build_activation_matrix, load_jsonl  # noqa: E402
from llama_50seed_span import load_direction  # noqa: E402  (same loader as the paper numbers)

LAYER, K, N_DRAW = 16, 229, 200
rng = np.random.default_rng(0)
unit = lambda v: v / np.linalg.norm(v)


def share(Q, d):
    d = unit(np.asarray(d, np.float64)); c = Q.T @ d
    return float(c @ c)


def orth(M):  # orthonormal basis of the column span of M [hidden, k]
    return np.linalg.qr(M.astype(np.float64))[0]


def stats(v):
    v = np.asarray(v); return {"mean": float(v.mean()), "std": float(v.std()),
                               "p95": float(np.percentile(v, 95)), "max": float(v.max()), "n": len(v)}


def main():
    # persona span, exactly as llama_50seed_span.py
    V = pickle.load(open(REPO / "full_trait_output/trait_detector.pkl", "rb"))["trait_matrix"]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    Qp = orth(V.T); H = Qp.shape[0]; assert Qp.shape[1] == K

    # targets
    ref = load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt")
    hb = [r for r in load_jsonl(REPO / "full_trait_output/harmbench_activations/classified_responses.jsonl")
          if r.get("attack_type") == "human_jailbreak"]
    acts = torch.load(REPO / "full_trait_output/harmbench_activations/activations.pt", map_location="cpu", weights_only=False)
    Xhb, yhb, _ = build_activation_matrix(hb, acts, LAYER); Xhb = Xhb.astype(np.float64)
    dmu = unit(Xhb[yhb == 1].mean(0) - Xhb[yhb == 0].mean(0))
    targets = {"refusal": ref, "dmu": dmu}

    # Alpaca / harmful plain activations from the v2 set
    rows = [json.loads(l) for l in open(REPO / "full_trait_output/roleplay_check/synthetic_v2_prompts.jsonl")]
    v2 = torch.load(REPO / "full_trait_output/roleplay_check/synthetic_v2_acts.pt", map_location="cpu")
    pick = lambda f: np.stack([v2[r["id"]].float().numpy() for r in rows if f(r)]).astype(np.float64)
    A = pick(lambda r: r["source"] == "alpaca" and r["frame"] == "plain")
    Hm = pick(lambda r: r["harm"] == "harmful" and r["frame"] == "plain")
    print(f"HB {Xhb.shape} (jb {yhb.mean():.3f}), Alpaca {A.shape}, harmful {Hm.shape}", flush=True)

    mu_a = A.mean(0)
    Ac = A - mu_a
    Qpca = np.linalg.svd(Ac.T, full_matrices=False)[0][:, :K]
    ref_rebuilt = unit(Hm.mean(0) - mu_a)

    res = {"n": {"hb": len(yhb), "hb_jb": int(yhb.sum()), "alpaca": len(A), "harmful": len(Hm)},
           "random_expectation": K / H,
           "check": {"cos(rebuilt harmful-minus-alpaca, arditi refusal)": float(ref_rebuilt @ ref),
                     "cos(refusal, dmu)": float(ref @ dmu)}}

    # A. subspaces
    rand_sub = [orth(rng.standard_normal((H, K))) for _ in range(N_DRAW)]
    act_sub = [orth(A[rng.choice(len(A), K, replace=False)].T) for _ in range(N_DRAW)]
    A_tab = {}
    for n, d in {**targets, "refusal_rebuilt": ref_rebuilt}.items():
        A_tab[n] = {"persona": share(Qp, d), "alpaca_pca229": share(Qpca, d),
                    "random229": stats([share(Q, d) for Q in rand_sub]),
                    "alpaca_acts229": stats([share(Q, d) for Q in act_sub])}
    res["A_subspaces"] = A_tab

    # B. directions in persona span
    def shuffled(X, y):
        out = []
        for _ in range(N_DRAW):
            p = rng.permutation(y); out.append(share(Qp, X[p == 1].mean(0) - X[p == 0].mean(0)))
        return stats(out)
    XR = np.vstack([Hm, A]); yR = np.r_[np.ones(len(Hm), int), np.zeros(len(A), int)]
    res["B_directions_in_persona"] = {
        "refusal": share(Qp, ref), "refusal_rebuilt": share(Qp, ref_rebuilt), "dmu": share(Qp, dmu),
        "alpaca_mean": share(Qp, mu_a),
        "hb_mean": share(Qp, Xhb.mean(0)),
        "random_gaussian": stats([share(Qp, rng.standard_normal(H)) for _ in range(N_DRAW)]),
        "shuffled_refusal": shuffled(XR, yR),
        "shuffled_dmu": shuffled(Xhb, yhb),
        "alpaca_mean_in_random229": stats([share(Q, mu_a) for Q in rand_sub]),
        "alpaca_mean_in_alpaca_pca229": share(Qpca, mu_a)}

    # C. centered Alpaca variance held by each subspace
    tot = float((Ac ** 2).sum())
    vs = lambda Q: float(((Ac @ Q) ** 2).sum() / tot)
    res["C_alpaca_variance_share"] = {"persona": vs(Qp), "alpaca_pca229": vs(Qpca),
                                      "random229": stats([vs(Q) for Q in rand_sub[:50]])}

    json.dump(res, open(HERE / "results.json", "w"), indent=1)
    f = lambda s: f"{s:.3f}" if isinstance(s, float) else f"{s['mean']:.3f} ± {s['std']:.3f} (max {s['max']:.3f})"
    L = [f"checks: {json.dumps({k: round(v, 3) for k, v in res['check'].items()})}", "",
         "A. share of target direction in each 229-dim subspace",
         "| target | persona span | random 229 | Alpaca PCA-229 | span of 229 Alpaca acts |", "|---|---|---|---|---|"]
    L += [f"| {n} | {f(t['persona'])} | {f(t['random229'])} | {f(t['alpaca_pca229'])} | {f(t['alpaca_acts229'])} |"
          for n, t in A_tab.items()]
    L += ["", "B. share in persona span of each direction", "| direction | share |", "|---|---|"]
    L += [f"| {n} | {f(v)} |" for n, v in res["B_directions_in_persona"].items()]
    L += ["", "C. share of centered Alpaca activation variance", "| subspace | share |", "|---|---|"]
    L += [f"| {n} | {f(v)} |" for n, v in res["C_alpaca_variance_share"].items()]
    open(HERE / "tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
