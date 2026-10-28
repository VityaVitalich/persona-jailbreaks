"""Gaussian control vs the raw-activation probe, OLMo-3-7B-Instruct next to Llama-3.1-8B and Qwen3-32B.
OLMo Gaussian: StandardScaler + LR on projections onto K=240 unit random directions (3 draws; no trait vectors exist for
OLMo, 240 = the Qwen trait count), C tuned on a behaviour-grouped validation split of the training set exactly as in
olmo_probe.py (layer 16, last prompt token). OLMo raw-probe scores come from olmo_probe's saved npz; Llama / Qwen raw and
Gaussian scores from qwen3_32b/transfer/diag/scores_cross_family_tuned.npz (no refits).
All three models are scored on the same test families (those OLMo has; PAIR is dropped while its OLMo run is pending).
Cluster bootstrap over HarmBench behaviours (B=1000, shared across test sets, orientation fixed at the point estimate);
paired delta raw - Gaussian(mean of draws), Bonferroni over (training sets + avg) x 3 models.
Saves OLMo Gaussian val/test scores and every bootstrap draw to scores_olmo_gaussian{TAG}.npz; table to olmo_gaussian{TAG}.json."""
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import olmo_probe_lib as L
K, G = 240, 3
TAG = L.TAG; SETS, FAM = L.SETS, L.FAM
D = {s: L.load(s) for s in SETS}; li = D["hb"][3].index(L.MAIN)
X = {s: D[s][0][:, li].astype(np.float64) for s in SETS}
Gs = [np.random.default_rng(g).standard_normal((K, X["hb"].shape[1])) for g in range(G)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
jobs = [delayed(L.cell)((g, a), a, X[a] @ Gs[g].T, D[a][1], D[a][2], {b: X[b] @ Gs[g].T for b in SETS if b != a}) for g in range(G) for a in SETS]
GO = dict(Parallel(n_jobs=8, backend="loky")(jobs)); print("olmo gaussian fits done", flush=True)
RZ = np.load(f"scores_olmo_probe{TAG}.npz")
raw_test = lambda a, b: RZ[[k for k in RZ.files if k.startswith(f"L{L.MAIN}__train_{a}__test_{b}__C")][0]]
# per model: labels, behaviour ids, test scores {det: {(a, b): s}}
M = {"olmo": {"y": {s: D[s][1] for s in SETS}, "beh": {s: np.array([r["behavior_id"] for r in D[s][2]]) for s in SETS},
              "raw": {(a, b): raw_test(a, b) for a in SETS for b in SETS if a != b},
              **{f"Gaussian{g}": {(a, b): GO[((g, a), a)]["test"][b] for a in SETS for b in SETS if a != b} for g in range(G)}}}
CZ = np.load("/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b/transfer/diag/scores_cross_family_tuned.npz")
def cz(m, det, a, b): return CZ[[k for k in CZ.files if k.startswith(f"{m}__{det}__train_{a}__test_{b}__C")][0]]
for m in ("llama", "qwen"):
    M[m] = {"y": {s: CZ[f"{m}__labels_{s}"] for s in SETS}, "beh": {s: CZ[f"{m}__behavior_ids_{s}"] for s in SETS},
            "raw": {(a, b): cz(m, "fullprobe", a, b) for a in SETS for b in SETS if a != b},
            **{f"Gaussian{g}": {(a, b): cz(m, f"Gaussian{g}", a, b) for a in SETS for b in SETS if a != b} for g in range(G)}}
B = 1000; NB = (len(SETS) + 1) * 3; zq = 1 - 0.05 / NB / 2; res, SC = {"K": K, "sets": SETS, "B": B, "bonferroni_m": NB, "cells": {}}, {}
for m, V in M.items():
    behs = sorted({b for s in SETS for b in V["beh"][s]}); bi = {b: i for i, b in enumerate(behs)}
    W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B); bidx = {s: np.array([bi[b] for b in V["beh"][s]]) for s in SETS}
    def boot(y, s, bo):
        sign = 1 if roc_auc_score(y, s) >= 0.5 else -1
        return roc_auc_score(y, sign * s), np.array([roc_auc_score(y, sign * s, sample_weight=W[k][bo]) if (W[k][bo] * y).sum() > 0 and (W[k][bo] * (1 - y)).sum() > 0 else np.nan for k in range(B)])
    T = {}
    for det in ["raw"] + [f"Gaussian{g}" for g in range(G)]:
        for a in SETS:
            pts, bss = zip(*[boot(V["y"][f], V[det][(a, f)], bidx[f]) for f in FAM if f != a]); T[(det, a)] = (np.mean(pts), np.nanmean(np.stack(bss), 0))
        T[(det, "avg")] = (np.mean([T[(det, a)][0] for a in SETS]), np.mean([T[(det, a)][1] for a in SETS], 0))
    for a in SETS + ["avg"]:
        gp = np.mean([T[(f"Gaussian{g}", a)][0] for g in range(G)]); gb = np.mean([T[(f"Gaussian{g}", a)][1] for g in range(G)], 0); rp, rb = T[("raw", a)]
        ci = lambda b: [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
        res["cells"][f"{m}__{a}"] = {"raw": float(rp), "raw_ci95": ci(rb), "gaussian": float(gp), "gaussian_ci95": ci(gb), "delta": float(rp - gp),
                                     "delta_ci_bonf": [float(np.nanpercentile(rb - gb, 100 * (1 - zq))), float(np.nanpercentile(rb - gb, 100 * zq))]}
        SC[f"boot__{m}__raw__{a}"] = rb.astype(np.float32); SC[f"boot__{m}__gaussian__{a}"] = gb.astype(np.float32)
for (g, a), v in GO.items():
    SC[f"olmo__Gaussian{g}__train_{a}__val_idx"] = v["val_idx"]
    for C, s in v["val_scores"].items(): SC[f"olmo__Gaussian{g}__train_{a}__val_C{C:g}"] = s
    for b, s in v["test"].items(): SC[f"olmo__Gaussian{g}__train_{a}__test_{b}__C{v['C']:g}"] = s
res["olmo_gaussian_C"] = {f"{g}__{a}": GO[((g, a), a)]["C"] for g in range(G) for a in SETS}
json.dump(res, open(f"olmo_gaussian{TAG}.json", "w"), indent=1); np.savez(f"scores_olmo_gaussian{TAG}.npz", **SC)
for m in M:
    for k in ("raw", "gaussian"): print(f"{m:5s} {k:8s} " + " ".join(f"{a}:{res['cells'][f'{m}__{a}'][k]:.3f}" for a in SETS + ["avg"]))
    print(f"{m:5s} delta    " + " ".join(f"{a}:{res['cells'][f'{m}__{a}']['delta']:+.3f}[{res['cells'][f'{m}__{a}']['delta_ci_bonf'][0]:+.3f},{res['cells'][f'{m}__{a}']['delta_ci_bonf'][1]:+.3f}]" for a in SETS + ["avg"]))
