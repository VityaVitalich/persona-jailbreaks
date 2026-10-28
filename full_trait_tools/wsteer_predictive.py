"""How predictive is w_steer (persona) -- the vector the steering attack steers with?

The vector is rebuilt with Dominic's recipe, verbatim from
  jailbreaking_tools/trait_steering_attack_jbb_persona_w.py  (train_source ->
  build_proj_features -> fit_classifier -> persona_w), clf_source="harmbench",
which is the run behind the paper's steer-w (persona) row (63% JBB ASR):

  rows   = harmbench_activations/classified_responses.jsonl, attack_type ==
           "human_jailbreak"                                       (n = 3180)
  X      = act[layer 16] @ trait_matrix.T, float32                 (229 traits)
  y      = row["jailbroken"]
  scaler = StandardScaler().fit(X)
  clf    = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(...)
  w_act  = (clf.coef_[0] / scaler.scale_) @ trait_matrix
  w      = w_act / ||w_act||

The identity check is the 5-fold CV AUC: the published run recorded
0.9681309216192938 in its summary.json, so reproducing that certifies this is
the same vector and not a lookalike.

Predictiveness is then measured the way tables/detection_llama.tex measures it
(run_all_traits_sweep_v2.safe_auc = max(auc, 1-auc), labels from each family's
responses.jsonl), on the five families the classifier never saw. Because the
classifier's decision function is w_act . x - const, the AUC of the raw
projection x . w IS the classifier's AUC -- no gap between "the direction we
steer with" and "the classifier it came from".

Reported per family: signed AUC (does the steering direction point the same way
as jailbreak?) next to the flipped max(auc, 1-auc) the tables use.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_traits_sweep_v2 import (  # noqa: E402
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    split_by_pool,
)

LAYER = 16
FT = REPO / "full_trait_output"
# label files exactly as run_all_traits_sweep_v2's defaults name them
FAMILIES = [
    ("GCG",     FT / "gcg_activations/responses.jsonl"),
    ("PAIR",    FT / "pair_activations/responses.jsonl"),
    ("PAP",     FT / "pap_activations/responses.jsonl"),
    ("GPTFuzz", FT / "gptfuzz_activations/responses.jsonl"),
    ("PEZ",     FT / "pez_activations/responses.jsonl"),
    ("WJB",     FT / "wildjailbreak_activations/classified_responses.jsonl"),
]
PUBLISHED = {"HarmBench": 0.86, "GCG": 0.83, "PAIR": 0.80, "PAP": 0.63,
             "GPTFuzz": 0.71, "PEZ": 0.85}
CV_AUC_PUBLISHED = 0.9681309216192938


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / np.linalg.norm(v)


def signed_auc(y, s):
    return float(roc_auc_score(y, s))


def flip(a):
    return max(a, 1.0 - a)


def project_exact(x32, V32):
    """build_proj_features, verbatim: float32 per-row `act @ trait_matrix.T`,
    stacked and cast to float32. Row-at-a-time because a batched BLAS call can
    block the sum differently and the identity check reads to 1e-16."""
    return np.stack([act @ V32.T for act in x32]).astype(np.float32)


def fit_persona_w(x_raw, y, V):
    """Dominic's recipe, verbatim (fit_classifier + persona_w)."""
    X = (x_raw @ V.T).astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf.fit(Xs, y)
    w_act = (clf.coef_[0] / scaler.scale_) @ V
    return unit(w_act), X


def main():
    V = np.load(FT / "trait_matrix_layer16.npy").astype(np.float64)
    names = json.loads((FT / "trait_names_layer16.json").read_text())
    print(f"trait matrix {V.shape}  ({len(names)} names)")

    rows = [r for r in load_jsonl(FT / "harmbench_activations/classified_responses.jsonl")
            if r.get("attack_type") == "human_jailbreak"]
    acts = torch.load(FT / "harmbench_activations/activations.pt",
                      map_location="cpu", weights_only=False)
    x_hb, y_hb, valid = build_activation_matrix(rows, acts, LAYER)
    x_hb = x_hb.astype(np.float64)
    print(f"HarmBench human_jailbreak: n={len(y_hb)} jb_rate={y_hb.mean():.10f}")

    # ── identity check ────────────────────────────────────────────────────────
    w, _ = fit_persona_w(x_hb, y_hb, V)
    X_hb = project_exact(x_hb.astype(np.float32),
                         np.load(FT / "trait_matrix_layer16.npy"))
    cv = cross_val_score(
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, random_state=42, C=1.0)),
        X_hb, y_hb, cv=5, scoring="roc_auc").mean()
    print(f"cv_auc={cv:.16f}  published={CV_AUC_PUBLISHED:.16f}  "
          f"delta={abs(cv - CV_AUC_PUBLISHED):.2e}")

    saved = torch.load(FT / "steering_directions/steering_directions_layer16.pt",
                       map_location="cpu", weights_only=False)
    anchors = {k.replace("__full", ""): unit(v.numpy())
               for k, v in saved["vectors"].items()}
    print("cos(rebuilt w, saved w_persona) = "
          f"{float(w @ anchors['w_persona']):.10f}")
    print("cos(w_persona, dmu) = "
          f"{float(anchors['w_persona'] @ anchors['dmu']):.4f}   "
          f"cos(w_persona, w_raw) = "
          f"{float(anchors['w_persona'] @ anchors['w_raw']):.4f}")

    # ── families (never seen by the classifier) ───────────────────────────────
    fam = {}
    for name, rp in FAMILIES:
        a = torch.load(rp.parent / "activations.pt", map_location="cpu",
                       weights_only=False)
        x, yy, _ = build_activation_matrix(load_jsonl(rp), a, LAYER)
        fam[name] = (x.astype(np.float64), yy)
        print(f"  {name}: n={len(yy)} jb={yy.mean():.3f}")

    # ── the steering vector, in sample on HarmBench, held out on families ─────
    print(f"\n{'='*88}\nw_steer (persona), fit on ALL 3180 HarmBench pairs "
          f"(the vector actually steered with)\n{'='*88}")
    print(f"{'set':10s} {'n':>6s} {'jb':>6s} {'signed AUC':>11s} "
          f"{'max(a,1-a)':>11s}   {'published detector':>18s}")
    a = signed_auc(y_hb, x_hb @ w)
    print(f"{'HarmBench':10s} {len(y_hb):6d} {y_hb.mean():6.3f} {a:11.3f} "
          f"{flip(a):11.3f}   {'(in sample)':>18s}")
    full = {}
    for name, (x, yy) in fam.items():
        a = signed_auc(yy, x @ w)
        full[name] = a
        pub = f"{PUBLISHED[name]:.2f}" if name in PUBLISHED else "-"
        print(f"{name:10s} {len(yy):6d} {yy.mean():6.3f} {a:11.3f} "
              f"{flip(a):11.3f}   {pub:>18s}")
    five = [full[f] for f in ("GCG", "PAIR", "PAP", "GPTFuzz", "PEZ")]
    print(f"{'AvgXfer':10s} {'':6s} {'':6s} {statistics.fmean(five):11.3f} "
          f"{statistics.fmean(flip(v) for v in five):11.3f}   {0.76:18.2f}")

    # ── same recipe, refit on the strict train pool, 50 seeds ────────────────
    print(f"\n{'='*88}\nsame recipe refit on the 70% behavior x template train pool, "
          f"50 seeds (clean HarmBench test)\n{'='*88}")
    per_seed = {k: [] for k in ["HarmBench"] + [f for f, _ in FAMILIES]}
    per_seed["AvgXfer"] = []
    for seed in range(50):
        tr_b, tr_t, te_b, te_t = get_pool_split(valid, 0.7, seed)
        tr, te = split_by_pool(valid, tr_b, tr_t, te_b, te_t)
        ws, _ = fit_persona_w(x_hb[tr], y_hb[tr], V)
        per_seed["HarmBench"].append(signed_auc(y_hb[te], x_hb[te] @ ws))
        for name, (x, yy) in fam.items():
            per_seed[name].append(signed_auc(yy, x @ ws))
        per_seed["AvgXfer"].append(statistics.fmean(
            per_seed[f][-1] for f in ("GCG", "PAIR", "PAP", "GPTFuzz", "PEZ")))
        if seed == 0:
            print(f"  seed 0: train={len(tr)} test={len(te)} "
                  f"(test jb {int(y_hb[te].sum())})")

    print(f"\n{'set':10s} {'signed mean':>12s} {'95% CI':>9s} "
          f"{'flipped':>8s}   {'published detector':>18s}")
    for k in ["HarmBench"] + [f for f, _ in FAMILIES] + ["AvgXfer"]:
        v = per_seed[k]
        ci = 1.96 * statistics.stdev(v) / len(v) ** 0.5
        pub = (f"{PUBLISHED[k]:.2f}" if k in PUBLISHED
               else "0.76" if k == "AvgXfer" else "-")
        print(f"{k:10s} {statistics.fmean(v):12.3f} {ci:9.3f} "
              f"{statistics.fmean(flip(x) for x in v):8.3f}   {pub:>18s}")

    # ── reference: the other steering arms on the same families ──────────────
    print(f"\n{'='*88}\nthe other steered directions, same families (signed AUC)"
          f"\n{'='*88}")
    print(f"{'direction':20s} " + "".join(f"{f:>9s}" for f, _ in FAMILIES)
          + f"{'AvgXfer5':>10s}{'HarmBench':>10s}")
    for name in ["w_persona", "w_raw", "dmu", "refusal_pos1_neg", "refusal_pos5_neg"]:
        d = anchors[name]
        vals = {f: signed_auc(fam[f][1], fam[f][0] @ d) for f, _ in FAMILIES}
        avg5 = statistics.fmean(vals[f] for f in ("GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"))
        print(f"{name:20s} " + "".join(f"{vals[f]:9.3f}" for f, _ in FAMILIES)
              + f"{avg5:10.3f}{signed_auc(y_hb, x_hb @ d):10.3f}")

    out = REPO / "full_trait_output/paper_figs_inputs/wsteer_predictive.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"cv_auc": cv, "cv_auc_published": CV_AUC_PUBLISHED,
         "n": int(len(y_hb)), "jb_rate": float(y_hb.mean()),
         "full_pool_fit": {"HarmBench_insample": signed_auc(y_hb, x_hb @ w), **full},
         "split_refit_50seeds": {k: {"mean": statistics.fmean(v),
                                     "ci95": 1.96 * statistics.stdev(v) / len(v) ** 0.5,
                                     "all": v} for k, v in per_seed.items()}},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
