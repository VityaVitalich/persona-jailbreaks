"""Raw-activation probe vs Gaussian control, per training set, for Llama-3.1-8B, Gemma-4-31B, Qwen3-32B and OLMo-3-7B.
Protocol (all models): StandardScaler + LR; C chosen on a behaviour-grouped validation split of the TRAINING set (HB also
template-disjoint), refit on the whole training set; transfer for a training set = mean folded AUC over the attack
families GCG/PAIR/PAP/GPTFuzz/PEZ other than itself. Gaussian = LR on projections onto K unit random directions (3 draws;
K = trait count: Llama 229, Gemma 240, Qwen 240, OLMo 240 since OLMo has no trait vectors).
Readouts: Llama L16, Gemma L30 (paper layers), Qwen L48, OLMo L16; last prompt token.
Fits: Gemma and OLMo here (olmo_probe_lib.cell); Llama / Qwen from qwen3_32b/transfer/diag/scores_cross_family_tuned.npz.
Cluster bootstrap over HarmBench behaviours (B=1000 per model, shared across test sets, orientation fixed at the point
estimate); paired delta raw - Gaussian(mean of draws), Bonferroni over (7 training sets + avg) x 4 models.
Saves Gemma / OLMo val and test scores for every fit and all bootstrap draws to scores_four_models.npz; table four_models.json."""
import json, os, numpy as np, torch
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import olmo_probe_lib as L
assert not L.MISSING, f"OLMo sets missing: {L.MISSING}"
SETS, FAM, G = L.SETS, L.FAM, 3
from olmo_probe_lib import gemma, olmo  # noqa: E402
M, SC = {}, {}
for m, loader, K in (("gemma", gemma, 240), ("olmo", olmo, 240)):
    D = {s: loader(s) for s in SETS}
    Gs = [np.random.default_rng(g).standard_normal((K, D["hb"][0].shape[1])) for g in range(G)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
    dets = [("raw", None)] + [(f"Gaussian{g}", V) for g, V in enumerate(Gs)]
    jobs = []
    for det, V in dets:
        F = {s: D[s][0] if V is None else D[s][0] @ V.T for s in SETS}
        jobs += [delayed(L.cell)((det, a), a, F[a], D[a][1], D[a][2], {b: F[b] for b in SETS if b != a}) for a in SETS]
    OUT = dict(Parallel(n_jobs=8, backend="loky")(jobs)); print(m, "fits done", flush=True)
    M[m] = {"y": {s: D[s][1] for s in SETS}, "beh": {s: np.array([r["behavior_id"] for r in D[s][2]]) for s in SETS},
            "C": {f"{det}__{a}": OUT[((det, a), a)]["C"] for det, _ in dets for a in SETS},
            **{det: {(a, b): OUT[((det, a), a)]["test"][b] for a in SETS for b in SETS if a != b} for det, _ in dets}}
    for ((det, a), _), v in OUT.items():
        p = f"{m}__{det}__train_{a}"; SC[f"{p}__val_idx"] = v["val_idx"]
        for C, s in v["val_scores"].items(): SC[f"{p}__val_C{C:g}"] = s
        for b, s in v["test"].items(): SC[f"{p}__test_{b}__C{v['C']:g}"] = s
    for s in SETS: SC[f"{m}__labels_{s}"] = D[s][1]; SC[f"{m}__behavior_ids_{s}"] = M[m]["beh"][s]; SC[f"{m}__pair_ids_{s}"] = np.array([r["pair_id"] for r in D[s][2]])
CZ = np.load("/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b/transfer/diag/scores_cross_family_tuned.npz")
def cz(m, det, a, b): return CZ[[k for k in CZ.files if k.startswith(f"{m}__{det}__train_{a}__test_{b}__C")][0]]
def czC(m, det, a): return float([k for k in CZ.files if k.startswith(f"{m}__{det}__train_{a}__test_")][0].rsplit("__C", 1)[1])
for m in ("llama", "qwen"):
    M[m] = {"y": {s: CZ[f"{m}__labels_{s}"] for s in SETS}, "beh": {s: CZ[f"{m}__behavior_ids_{s}"] for s in SETS},
            "C": {f"{det}__{a}": czC(m, "fullprobe" if det == "raw" else det, a) for det in ["raw"] + [f"Gaussian{g}" for g in range(G)] for a in SETS},
            "raw": {(a, b): cz(m, "fullprobe", a, b) for a in SETS for b in SETS if a != b},
            **{f"Gaussian{g}": {(a, b): cz(m, f"Gaussian{g}", a, b) for a in SETS for b in SETS if a != b} for g in range(G)}}
B = 1000; NB = (len(SETS) + 1) * len(M); zq = 1 - 0.05 / NB / 2
res = {"sets": SETS, "B": B, "bonferroni_m": NB, "cells": {}, "per_family": {}, "C": {m: M[m]["C"] for m in M}}
ci = lambda b: [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
for m, V in M.items():
    behs = sorted({b for s in SETS for b in V["beh"][s]}); bi = {b: i for i, b in enumerate(behs)}
    W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B); bidx = {s: np.array([bi[b] for b in V["beh"][s]]) for s in SETS}
    def boot(y, s, bo):
        sign = 1 if roc_auc_score(y, s) >= 0.5 else -1
        return roc_auc_score(y, sign * s), np.array([roc_auc_score(y, sign * s, sample_weight=W[k][bo]) if (W[k][bo] * y).sum() > 0 and (W[k][bo] * (1 - y)).sum() > 0 else np.nan for k in range(B)])
    T = {}
    for det in ["raw"] + [f"Gaussian{g}" for g in range(G)]:
        for a in SETS:
            fams = [f for f in FAM if f != a]; pts, bss = zip(*[boot(V["y"][f], V[det][(a, f)], bidx[f]) for f in fams])
            T[(det, a)] = (np.mean(pts), np.nanmean(np.stack(bss), 0)); res["per_family"][f"{m}__{det}__{a}"] = dict(zip(fams, map(float, pts)))
        T[(det, "avg")] = (np.mean([T[(det, a)][0] for a in SETS]), np.mean([T[(det, a)][1] for a in SETS], 0))
    for a in SETS + ["avg"]:
        gp = np.mean([T[(f"Gaussian{g}", a)][0] for g in range(G)]); gb = np.mean([T[(f"Gaussian{g}", a)][1] for g in range(G)], 0); rp, rb = T[("raw", a)]
        res["cells"][f"{m}__{a}"] = {"raw": float(rp), "raw_ci95": ci(rb), "gaussian": float(gp), "gaussian_ci95": ci(gb), "delta": float(rp - gp),
                                     "delta_ci_bonf": [float(np.nanpercentile(rb - gb, 100 * (1 - zq))), float(np.nanpercentile(rb - gb, 100 * zq))]}
        SC[f"boot__{m}__raw__{a}"] = rb.astype(np.float32); SC[f"boot__{m}__gaussian__{a}"] = gb.astype(np.float32)
json.dump(res, open("four_models.json", "w"), indent=1); np.savez("scores_four_models.npz", **SC)
for m in M:
    for k in ("raw", "gaussian"): print(f"{m:5s} {k:8s} " + " ".join(f"{a}:{res['cells'][f'{m}__{a}'][k]:.3f}" for a in SETS + ["avg"]))
    print(f"{m:5s} delta    " + " ".join(f"{a}:{res['cells'][f'{m}__{a}']['delta']:+.3f}[{res['cells'][f'{m}__{a}']['delta_ci_bonf'][0]:+.3f},{res['cells'][f'{m}__{a}']['delta_ci_bonf'][1]:+.3f}]" for a in SETS + ["avg"]))
