#!/usr/bin/env python3
"""How much does the steer-traits selection heuristic change if the standardizer
is respected?  The scripts rank traits by  proj_raw[i] * coef_std[i]; the actual
per-trait contribution to the logit is  coef_std[i] * (proj_raw[i]-mean[i])/scale[i]
 =  coef_proj[i] * (proj_raw[i] - mean[i]).
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.home() / "persona-vector-jbing"
sys.path.insert(0, str(REPO / "full_trait_tools"))
os.chdir(REPO)
from run_all_traits_sweep_v2 import build_activation_matrix, load_jsonl  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

LAYER = 16
rows = [r for r in load_jsonl(REPO / "full_trait_output/harmbench_activations/classified_responses.jsonl")
        if r.get("attack_type") == "human_jailbreak"]
acts = torch.load(REPO / "full_trait_output/harmbench_activations/activations.pt",
                  map_location="cpu", weights_only=False)
x_raw, y, valid = build_activation_matrix(rows, acts, LAYER)
best = pickle.load(open(REPO / "full_trait_output/all_traits_sweep_v2/best_model.pkl", "rb"))
V = best["trait_matrix"].astype(np.float64)
names = best["trait_names"]
X = x_raw.astype(np.float64) @ V.T

# the steering scripts' own recipe: fit_classifier (C=1.0)
sc = StandardScaler().fit(X)
clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0).fit(sc.transform(X), y)
coef_std = clf.coef_[0]
coef_proj = coef_std / sc.scale_

comp = coef_std > 0
refu = coef_std < 0
print(f"{comp.sum()} compliance / {refu.sum()} refusal traits "
      f"(sign partition is scale-invariant, so identical either way)")

for K in (3, 5):
    ov_inc, ov_dec = [], []
    for i in range(len(X)):
        proj = X[i]
        w_code = proj * coef_std                      # what the code ranks by
        w_true = coef_proj * (proj - sc.mean_)        # actual logit contribution
        for mask, rev, store in ((comp, False, ov_inc), (refu, True, ov_dec)):
            idx = np.flatnonzero(mask)
            a = idx[np.argsort(w_code[idx])][::-1][:K] if rev else idx[np.argsort(w_code[idx])][:K]
            b = idx[np.argsort(w_true[idx])][::-1][:K] if rev else idx[np.argsort(w_true[idx])][:K]
            store.append(len(set(a) & set(b)) / K)
    print(f"  K={K}: increase-list overlap {np.mean(ov_inc):.2f}, "
          f"decrease-list overlap {np.mean(ov_dec):.2f}")

# also: the fixed steering vector each ranking builds, per example
def build(idx_inc, idx_dec):
    v = V[idx_inc].sum(0) - V[idx_dec].sum(0)
    return v / np.linalg.norm(v)

cos = []
for i in range(0, len(X), 7):
    proj = X[i]
    w_code = proj * coef_std
    w_true = coef_proj * (proj - sc.mean_)
    ci = np.flatnonzero(comp); cd = np.flatnonzero(refu)
    a = build(ci[np.argsort(w_code[ci])][:3], cd[np.argsort(w_code[cd])][::-1][:3])
    b = build(ci[np.argsort(w_true[ci])][:3], cd[np.argsort(w_true[cd])][::-1][:3])
    cos.append(float(np.dot(a, b)))
print(f"  cos(steer-traits vector code vs std-consistent), k=3: "
      f"mean={np.mean(cos):.3f} min={np.min(cos):.3f}")
