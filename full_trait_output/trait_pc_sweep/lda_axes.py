#!/usr/bin/env python3
"""Persona-discriminant axes (multi-class LDA over the 229 Llama traits, layer 16), label-free w.r.t. jailbreaks.

Per trait t the stored vector is T_t = unit(mean_s d_ts), d_ts = h(positive_p_q) - h(negative_p_q), 200 samples
(5 prompt pairs x 40 questions). Between-persona scatter S_B = centered T, exactly the PCA's input. Within-persona
scatter S_W = pooled covariance of the residuals (d_ts - mean_t d_t) / |mean_t d_t| (same scale as the unit T_t).
Axes solve S_B v = lam S_W(a) v with shrinkage S_W(a) = (1-a) S_W + a tr(S_W)/d I; a = 1 is the PCA of T.

Writes lda_bases_llama.npz: {f"lda_a{a}": [228 x 4096] unit-norm rows, ordered by lam} and runs a label-free
sanity check: nearest-centroid identification of held-out persona samples (questions 30-39; axes, centroids and
S_W refit on questions 0-29) in the top-k axes, PCA vs LDA."""
import json, re, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.covariance import ledoit_wolf_shrinkage

OUT = Path.home() / "persona-vector-jbing/full_trait_output"; HERE = OUT / "trait_pc_sweep"; L = 16
names = json.load(open(OUT / "trait_names_layer16.json")); T = np.load(OUT / "trait_matrix_layer16.npy").astype(np.float64)
T /= np.linalg.norm(T, axis=1, keepdims=True); n, d = T.shape
t0 = time.time(); D, Q = [], []
for nm in names:
    a = torch.load(OUT / f"traits40_activations/pre_generation_last_token/{nm}.pt", map_location="cpu", weights_only=False)
    ks = sorted(k[9:] for k in a if k.startswith("positive_") and "negative_" + k[9:] in a)
    D.append(np.stack([(a["positive_" + k][L] - a["negative_" + k][L]).float().numpy() for k in ks]).astype(np.float64))
    Q.append(np.array([int(re.search(r"_q(\d+)", k).group(1)) for k in ks]))
print(f"loaded {n} traits in {time.time()-t0:.0f}s; samples per trait {sorted({len(x) for x in D})}", flush=True)
assert min(D[i].mean(0) @ T[i] / np.linalg.norm(D[i].mean(0)) for i in range(n)) > 0.999  # stored vector = mean diff

def axes(means, resid, alphas):
    """means: n x d class means (rows), resid: list of per-class residual arrays (already scaled). Returns {a: V}."""
    R = np.vstack(resid); Sw = R.T @ R / (len(R) - len(resid)); mu = np.trace(Sw) / d
    Mc = means - means.mean(0); out = {}
    alphas = dict(alphas)
    if "lw" in alphas: alphas["lw"] = float(ledoit_wolf_shrinkage(R, assume_centered=True)); print(f"  Ledoit-Wolf alpha = {alphas['lw']:.4f}", flush=True)
    for tag, a in alphas.items():
        lam, U = np.linalg.eigh((1 - a) * Sw + a * mu * np.eye(d))
        W = (U / np.sqrt(lam)) @ U.T                     # S_W(a)^{-1/2}
        _, s, E = np.linalg.svd(Mc @ W, full_matrices=False)
        r = int((s > 1e-8 * s[0]).sum()); V = (W @ E[:r].T).T
        out[tag] = (V / np.linalg.norm(V, axis=1, keepdims=True), W @ E[:r].T, a)  # unit axes; unnormalized (whitened) for the check
    return out

ALPHAS = {"a0.001": 0.001, "a0.01": 0.01, "a0.1": 0.1, "a0.3": 0.3, "lw": None, "a1(pca)": 1.0}

# ---- label-free sanity check: identify held-out persona samples
tr = [x[q < 30] for x, q in zip(D, Q)]; te = [x[q >= 30] for x, q in zip(D, Q)]
m_tr = np.stack([x.mean(0) for x in tr]); sc = 1 / np.linalg.norm(m_tr, axis=1)
ax = axes(m_tr * sc[:, None], [(x - x.mean(0)) * s for x, s in zip(tr, sc)], ALPHAS)
Xte = np.vstack([x * s for x, s in zip(te, sc)]); yte = np.concatenate([[i] * len(x) for i, x in enumerate(te)])
check = {}
print("held-out persona identification (nearest centroid among 229, chance 0.4%):")
for tag, (_, P, a) in ax.items():
    row = {}
    for k in (5, 10, 20, 50, 100, 228):
        Pk = P[:, :k]; C = (m_tr * sc[:, None]) @ Pk; Z = Xte @ Pk
        pred = np.argmin((C ** 2).sum(1)[None] - 2 * Z @ C.T, 1); row[k] = float((pred == yte).mean())
    check[tag] = {"alpha": a, "acc": row}
    print(f"  {tag:8s} (a={a:.4f}) " + " ".join(f"k={k}:{v:.3f}" for k, v in row.items()), flush=True)

# ---- final axes on all samples
m = np.stack([x.mean(0) for x in D]); sc = 1 / np.linalg.norm(m, axis=1)
ax = axes(T, [(x - x.mean(0)) * s for x, s in zip(D, sc)], ALPHAS)
_, _, Vp = np.linalg.svd(T - T.mean(0), full_matrices=False)
print(f"  check: a=1 axes vs PCA |cos| top5 = {np.abs(np.sum(ax['a1(pca)'][0][:5] * Vp[:5], 1)).round(4)}")
np.savez(HERE / "lda_bases_llama.npz", **{("lda_" + t.replace("(pca)", "")): v.astype(np.float32) for t, (v, _, _) in ax.items() if t != "a1(pca)"})
json.dump({"alphas": {t: a for t, (_, _, a) in ax.items()}, "persona_id_check": check}, open(HERE / "lda_axes_llama.json", "w"), indent=1)
print(f"done {time.time()-t0:.0f}s", flush=True)
