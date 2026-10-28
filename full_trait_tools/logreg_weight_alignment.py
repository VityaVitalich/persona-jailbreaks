#!/usr/bin/env python3
"""Stability + std-correction audit for the all-traits logistic regression.

Q1  How well do the learned weight vectors agree across the 50 pool splits
    (cosine), in trait space and mapped back into activation space?
Q2  Is the logreg trained on normalized projections?  (inspect the pipeline)
Q3  When the weights are mapped back onto the trait vectors for steering,
    does dropping the StandardScaler std change the resulting direction?
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.home() / "persona-vector-jbing"
sys.path.insert(0, str(REPO / "full_trait_tools"))
os.chdir(REPO)
from run_all_traits_sweep_v2 import (  # noqa: E402
    build_activation_matrix, get_pool_split, load_jsonl, safe_auc, split_by_pool,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = REPO / "results"
N_SEEDS, TRAIN_FRAC, LAYER = 50, 0.7, 16


def unit(v):
    return v / np.linalg.norm(v)


def pairwise_cos(M):
    """M: [n_vectors, dim] -> off-diagonal cosine stats."""
    U = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = U @ U.T
    iu = np.triu_indices(len(M), k=1)
    c = C[iu]
    return {"mean": float(c.mean()), "min": float(c.min()),
            "max": float(c.max()), "p05": float(np.percentile(c, 5))}


rows = [r for r in load_jsonl(REPO / "full_trait_output/harmbench_activations/classified_responses.jsonl")
        if r.get("attack_type") == "human_jailbreak"]
acts = torch.load(REPO / "full_trait_output/harmbench_activations/activations.pt",
                  map_location="cpu", weights_only=False)
x_raw, y, valid = build_activation_matrix(rows, acts, LAYER)

best = pickle.load(open(REPO / "full_trait_output/all_traits_sweep_v2/best_model.pkl", "rb"))
V = best["trait_matrix"].astype(np.float64)          # [229, 4096]
trait_names = best["trait_names"]
row_norms = np.linalg.norm(V, axis=1)
print(f"acts {x_raw.shape}  traits {V.shape}  jb rate {y.mean():.3f}")
print(f"trait rows unit-norm? min={row_norms.min():.6f} max={row_norms.max():.6f}")

X = x_raw.astype(np.float64) @ V.T                    # raw projections [N, 229]
print(f"raw projection feature std: min={X.std(0).min():.3f} "
      f"max={X.std(0).max():.3f} ratio={X.std(0).max()/X.std(0).min():.1f}")


def build_pipeline(seed):
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                                                max_iter=4000, class_weight=None,
                                                random_state=seed))])


coef_std_all, coef_proj_all, scales_all, aucs = [], [], [], []
for seed in range(N_SEEDS):
    tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(valid, TRAIN_FRAC, seed)
    tr, te = split_by_pool(valid, tr_beh, tr_tpl, te_beh, te_tpl)
    if not tr or not te or len(np.unique(y[tr])) < 2:
        continue
    pipe = build_pipeline(seed)
    pipe.fit(X[tr], y[tr])
    aucs.append(safe_auc(y[te], pipe.decision_function(X[te])))
    sc = pipe.named_steps["sc"]
    cs = pipe.named_steps["clf"].coef_.ravel().astype(np.float64)
    coef_std_all.append(cs)
    coef_proj_all.append(cs / sc.scale_)
    scales_all.append(sc.scale_.copy())

coef_std_all = np.stack(coef_std_all)
coef_proj_all = np.stack(coef_proj_all)
scales = np.stack(scales_all)
print(f"\nfitted {len(coef_std_all)} seeds, held-out AUC {np.mean(aucs):.4f} "
      f"+- {np.std(aucs):.4f}")

# ── Q1: cross-seed agreement ──────────────────────────────────────────────────
W_act_correct = coef_proj_all @ V                       # [seeds, 4096] std-corrected
W_act_naive = coef_std_all @ V                          # [seeds, 4096] no std division
res = {
    "trait_space_coef_std":  pairwise_cos(coef_std_all),
    "trait_space_coef_proj": pairwise_cos(coef_proj_all),
    "activation_space_correct": pairwise_cos(W_act_correct),
    "activation_space_naive":   pairwise_cos(W_act_naive),
}
print("\n=== Q1  cross-seed cosine agreement of the learned weight vector ===")
for k, v in res.items():
    print(f"  {k:26s} mean={v['mean']:+.4f}  min={v['min']:+.4f}  p05={v['p05']:+.4f}")

# mean direction and its own concentration
w_bar = unit(W_act_correct.mean(0))
cos_to_mean = (W_act_correct / np.linalg.norm(W_act_correct, axis=1, keepdims=True)) @ w_bar
print(f"  cos(seed w_act, mean w_act): mean={cos_to_mean.mean():.4f} "
      f"min={cos_to_mean.min():.4f}")

# ── Q3: does the std correction matter? ───────────────────────────────────────
cos_corr_vs_naive = [float(np.dot(unit(a), unit(b)))
                     for a, b in zip(W_act_correct, W_act_naive)]
cos_trait_space = [float(np.dot(unit(a), unit(b)))
                   for a, b in zip(coef_proj_all, coef_std_all)]
print("\n=== Q3  std-corrected vs naive mapping (same seed, same fit) ===")
print(f"  cos in trait space      : mean={np.mean(cos_trait_space):.4f} "
      f"min={np.min(cos_trait_space):.4f}")
print(f"  cos in activation space : mean={np.mean(cos_corr_vs_naive):.4f} "
      f"min={np.min(cos_corr_vs_naive):.4f}")
print(f"  scaler.scale_ across traits: min={scales.mean(0).min():.4f} "
      f"max={scales.mean(0).max():.4f} "
      f"ratio={scales.mean(0).max()/scales.mean(0).min():.1f}")

# how the ranking of "top traits" changes if std is ignored
c_std = np.abs(coef_std_all.mean(0))
c_proj = np.abs(coef_proj_all.mean(0))
top_std = set(np.argsort(-c_std)[:10])
top_proj = set(np.argsort(-c_proj)[:10])
print(f"  top-10 traits overlap (|coef_std| vs |coef_proj|): "
      f"{len(top_std & top_proj)}/10")
print(f"    by coef_std : {[trait_names[i] for i in np.argsort(-c_std)[:10]]}")
print(f"    by coef_proj: {[trait_names[i] for i in np.argsort(-c_proj)[:10]]}")

# ── steering-relevant: AUC of each mapped-back direction as a linear score ────
lin = {}
for tag, Wm in (("correct", W_act_correct), ("naive", W_act_naive)):
    a = []
    for i, seed in enumerate(range(len(Wm))):
        tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(valid, TRAIN_FRAC, seed)
        tr, te = split_by_pool(valid, tr_beh, tr_tpl, te_beh, te_tpl)
        a.append(safe_auc(y[te], x_raw[te].astype(np.float64) @ Wm[i]))
    lin[tag] = (float(np.mean(a)), float(np.std(a)))
print("\n  held-out AUC of the mapped-back activation direction (linear score):")
for k, (m, s) in lin.items():
    print(f"    {k:8s} {m:.4f} +- {s:.4f}")

# ── saved artefact sanity: is best_model.pkl's coef_proj std-corrected? ───────
sc_saved = best["pipeline"].named_steps["sc"]
chk = np.abs(best["coef_proj"] - best["coef_std"] / sc_saved.scale_).max()
print(f"\n  best_model.pkl: max|coef_proj - coef_std/scale_| = {chk:.2e} "
      f"(0 => stored std-corrected)")

json.dump({"cross_seed": res,
           "cos_to_mean": {"mean": float(cos_to_mean.mean()),
                           "min": float(cos_to_mean.min())},
           "corrected_vs_naive": {"trait_space_mean": float(np.mean(cos_trait_space)),
                                  "act_space_mean": float(np.mean(cos_corr_vs_naive)),
                                  "act_space_min": float(np.min(cos_corr_vs_naive))},
           "scale_ratio": float(scales.mean(0).max() / scales.mean(0).min()),
           "linear_auc": lin,
           "held_out_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))}},
          open(OUT / "logreg_weight_alignment.json", "w"), indent=1)
print("saved", OUT / "logreg_weight_alignment.json")
