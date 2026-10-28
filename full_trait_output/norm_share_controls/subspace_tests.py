#!/usr/bin/env python3
"""
Subspace tests for persona span vs matched 229-dim controls (control_bases_llama.npz), Llama L16.
Every subspace is used through an orthonormal basis Q of its span, so nothing depends on how the
229 directions are parameterized (unlike the StandardScaler + L2 detector of trait_pc_sweep.py).

  share    ||Q^T d||^2 for the refusal direction (Arditi, L16 last token) and dmu (all data)
  proj     held-out AUC of the projections Q Q^T d on HarmBench test, same 50 pool splits as the paper;
           refusal scored negated, dmu refit on each train split. Signed AUC, sign fixed a priori.
  lda      option 2, held-out Mahalanobis separability. On each train split, in subspace coordinates
           z = Q^T h: class means and pooled within-class covariance (Ledoit-Wolf shrinkage), w = S^-1 dmu.
           On the test split: d'^2 = (mean_jb - mean_ref)^2 / pooled within-class var of the scores z.w,
           and AUC. The same in the full 4096-dim space gives the reference; ratio = share of held-out
           linear separability the subspace keeps. Transfer: the HB-trained w scored on each family.
  harm     the same LDA ratio for harmful vs Alpaca requests (the refusal contrast): 897 harmful + 797
           Alpaca plain prompts from the v2 role-play set, 20 random 50/50 request splits.
Writes subspace_tests.json and subspace_tests.md. Saved activations only.
"""
import json, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious macOS Accelerate matmul warnings
from pathlib import Path
import numpy as np, torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
from llama_50seed_span import load_direction  # noqa: E402
from build_bases import alpaca_acts  # noqa: E402

FAMS = S.FAMS
unit = lambda v: v / np.linalg.norm(v)


def lda_fit(Z, y):
    m1, m0 = Z[y == 1].mean(0), Z[y == 0].mean(0)
    R = np.vstack([Z[y == 1] - m1, Z[y == 0] - m0])
    return np.linalg.solve(LedoitWolf(assume_centered=True).fit(R).covariance_, m1 - m0)


def dprime2(s, y):
    a, b = s[y == 1], s[y == 0]
    v = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return float((a.mean() - b.mean()) ** 2 / v)


def main():
    cfg = S.CFG["llama"]
    T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    bases = {"persona": T, **{k: v.astype(np.float64) for k, v in np.load(HERE / "control_bases_llama.npz").items()}}
    Q = {n: np.linalg.qr(V.T)[0] for n, V in bases.items()}
    Q["full"] = None

    hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    Xh, yh, hv = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
    Xh = Xh.astype(np.float64)
    fam = {}
    for f, (rp, ap) in cfg["fams"].items():
        X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)
        fam[f] = (X.astype(np.float64), y)
    ref = load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt")
    dmu_all = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))
    splits = [S.pool_split(hv, s) for s in range(50)]
    proj = lambda q, d: d if q is None else q @ (q.T @ d)
    coords = lambda q, X: X if q is None else X @ q

    res = {}
    for n, q in Q.items():
        r = {"share": {"refusal": float(np.sum((q.T @ ref) ** 2)) if q is not None else 1.0,
                       "dmu": float(np.sum((q.T @ dmu_all) ** 2)) if q is not None else 1.0},
             "proj_auc": {"refusal": [], "dmu": []}, "lda_d2": [], "lda_auc": [], "lda_xfer_auc": []}
        Zh = coords(q, Xh); Zf = {f: coords(q, X) for f, (X, _) in fam.items()}
        for tr, te in splits:
            tr, te = np.array(tr), np.array(te)
            d_tr = unit(Xh[tr][yh[tr] == 1].mean(0) - Xh[tr][yh[tr] == 0].mean(0))
            r["proj_auc"]["refusal"].append(roc_auc_score(yh[te], -Xh[te] @ proj(q, ref)))
            r["proj_auc"]["dmu"].append(roc_auc_score(yh[te], Xh[te] @ proj(q, d_tr)))
            w = lda_fit(Zh[tr], yh[tr]); s = Zh[te] @ w
            r["lda_d2"].append(dprime2(s, yh[te])); r["lda_auc"].append(roc_auc_score(yh[te], s))
            r["lda_xfer_auc"].append(np.mean([roc_auc_score(y, Zf[f] @ w) for f, (_, y) in fam.items()]))
        res[n] = r
        print(f"{n:11s} share ref {r['share']['refusal']:.3f} dmu {r['share']['dmu']:.3f} | proj AUC ref "
              f"{np.mean(r['proj_auc']['refusal']):.3f} dmu {np.mean(r['proj_auc']['dmu']):.3f} | LDA d'2 "
              f"{np.mean(r['lda_d2']):.3f} AUC {np.mean(r['lda_auc']):.3f} xfer {np.mean(r['lda_xfer_auc']):.3f}", flush=True)

    # harmful vs Alpaca (refusal contrast), 20 request splits
    rows = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
    v2 = torch.load(OUT / "roleplay_check/synthetic_v2_acts.pt", map_location="cpu")
    Hm = np.stack([v2[r["id"]].float().numpy() for r in rows if r["harm"] == "harmful" and r["frame"] == "plain"]).astype(np.float64)
    A = alpaca_acts(); XR = np.vstack([Hm, A]); yR = np.r_[np.ones(len(Hm), int), np.zeros(len(A), int)]
    rng = np.random.default_rng(0); hs = []
    for _ in range(20):
        p = rng.permutation(len(yR)); hs.append((p[: len(p) // 2], p[len(p) // 2:]))
    for n, q in Q.items():
        Z = coords(q, XR); d2 = []
        for tr, te in hs:
            d2.append(dprime2(Z[te] @ lda_fit(Z[tr], yR[tr]), yR[te]))
        res[n]["harm_d2"] = d2
    json.dump(res, open(HERE / "subspace_tests.json", "w"), indent=1)

    # table: means over splits; control families averaged over their 5 draws (min-max in brackets)
    groups = {"persona span": ["persona"], "Alpaca PCA-229": ["alpaca_pca"], "HB PCA-229 (unlabeled)": ["hb_pca"],
              "229 Alpaca activations": [f"acts_d{i}" for i in range(5)],
              "229 shuffled-label contrasts": [f"noise_d{i}" for i in range(5)],
              "random 229": [f"gauss_d{i}" for i in range(5)], "full space (4096)": ["full"]}
    fd2, fh2 = np.mean(res["full"]["lda_d2"]), np.mean(res["full"]["harm_d2"])
    cols = [("share refusal", lambda r: r["share"]["refusal"]), ("share Δμ", lambda r: r["share"]["dmu"]),
            ("proj AUC refusal", lambda r: np.mean(r["proj_auc"]["refusal"])), ("proj AUC Δμ", lambda r: np.mean(r["proj_auc"]["dmu"])),
            ("LDA d'² kept (HB)", lambda r: np.mean(r["lda_d2"]) / fd2), ("LDA AUC HB", lambda r: np.mean(r["lda_auc"])),
            ("LDA AUC transfer", lambda r: np.mean(r["lda_xfer_auc"])), ("LDA d'² kept (harmful vs Alpaca)", lambda r: np.mean(r["harm_d2"]) / fh2)]
    L = ["| subspace | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for g, ns in groups.items():
        cells = []
        for _, f in cols:
            v = np.array([f(res[n]) for n in ns])
            cells.append(f"{v.mean():.3f}" + (f" [{v.min():.3f}, {v.max():.3f}]" if len(v) > 1 else ""))
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L += ["", f"full-space held-out d'² HB {fd2:.2f}, harmful vs Alpaca {fh2:.2f}. 50 HB pool splits (20 for harmful vs Alpaca)."]
    open(HERE / "subspace_tests.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
