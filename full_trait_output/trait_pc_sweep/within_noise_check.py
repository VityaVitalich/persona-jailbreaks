#!/usr/bin/env python3
"""Where does the jailbreak shift sit relative to within-persona variation? Shares of dmu (HB jailbroken - refused) and
PersonaJB w in the top-k eigendirections of S_W (within-persona scatter of the trait diffs, LDA's denominator) vs the
top-k trait PCs (S_B, LDA's numerator) vs random k-dim subspaces. Also: how the HB prompts' own variance projects."""
import json, re, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
sys.path.insert(0, "."); import trait_pc_sweep as S
from pathlib import Path
OUT = Path.home() / "persona-vector-jbing/full_trait_output"
names = json.load(open(OUT / "trait_names_layer16.json")); T = np.load(OUT / "trait_matrix_layer16.npy").astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
R = []
for nm in names:
    a = torch.load(OUT / f"traits40_activations/pre_generation_last_token/{nm}.pt", map_location="cpu", weights_only=False)
    ks = [k[9:] for k in a if k.startswith("positive_")]
    D = np.stack([(a["positive_" + k][16] - a["negative_" + k][16]).float().numpy() for k in ks]).astype(np.float64)
    m = D.mean(0); R.append((D - m) / np.linalg.norm(m))
R = np.vstack(R); Sw = R.T @ R / len(R); lw, Uw = np.linalg.eigh(Sw); Uw = Uw[:, ::-1]; lw = lw[::-1]
_, _, Vp = np.linalg.svd(T - T.mean(0), full_matrices=False)
cfg = S.CFG["llama"]; rows = [x for x in S.load_jsonl(cfg["human"][0]) if x.get("attack_type") == "human_jailbreak"]
X, y, valid = S.build_matrix(rows, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16); X = X.astype(np.float64)
dmu = X[y == 1].mean(0) - X[y == 0].mean(0)
tr, _ = S.pool_split(valid, 0); mdl = S.model(0).fit(X[tr] @ T.T, y[tr]); w = T.T @ (mdl[-1].coef_[0] / mdl[0].scale_)
Q, _ = np.linalg.qr(T.T); dmu_in = Q @ (Q.T @ dmu)   # in-trait-span part, the part any trait-based detector can use
rng = np.random.default_rng(0)
def sh(U, v, k): return float(np.sum((U[:, :k].T @ v) ** 2) / np.sum(v ** 2))
res = {}
print(f"S_W: top-10 eigdirs hold {lw[:10].sum()/lw.sum():.2f} of within-persona variance, top-50 {lw[:50].sum()/lw.sum():.2f}")
print("share captured by top-k directions of …            k=   5    10    20    50   100")
for vn, v in [("dmu (full)", dmu), ("dmu in trait span", dmu_in), ("PersonaJB w", w)]:
    for bn, U in [("S_W (within-persona)", Uw), ("trait PCs (S_B)", Vp.T)]:
        r = [sh(U, v, k) for k in (5, 10, 20, 50, 100)]; res[f"{vn} | {bn}"] = r
        print(f"  {vn:18s} in {bn:22s}  " + " ".join(f"{x:5.2f}" for x in r))
    rr = np.mean([[sh(np.linalg.qr(rng.standard_normal((4096, 100)))[0], v, k) for k in (5, 10, 20, 50, 100)] for _ in range(5)], 0)
    print(f"  {vn:18s} in {'random':22s}  " + " ".join(f"{x:5.2f}" for x in rr))
# cosine between the two subspaces: how much of S_W's top-20 lies in span(T)?
print(f"top-20 S_W directions: mean share inside span(T) = {np.mean(np.sum((Q.T @ Uw[:, :20]) ** 2, 0)):.2f}")
json.dump(res, open("within_noise_check.json", "w"), indent=1)
