"""Can a persona-based detector match the raw-activation probe? Variants vs raw LR, single-set training, all four models at
their paper readouts (Llama L16, Gemma L30, Qwen L48, OLMo L16; last prompt token).
  PCA_k     LR on projections onto the top-k principal components of the (centered) trait matrix, k in KS
  top_k     LR on the k traits with the highest single-trait folded AUC on the TRAINING set (k = 1..10)
  rand_k    LR on k random traits (10 draws per k; reported as the mean over draws)
  PersonaJB all traits (reference); concat = [PersonaJB projections, raw activation] -> LR
Every detector: StandardScaler + LR, C from GRID chosen on a behaviour-grouped validation split of the training set (HB also
template-disjoint), refit on all training data; transfer per training set = mean folded AUC over the other attack families.
Raw LR scores come from earlier fits with the same protocol (scores_four_models.npz for Gemma / OLMo,
scores_cross_family_tuned.npz for Llama / Qwen); labels are asserted to be in the same row order.
Cluster bootstrap over HarmBench behaviours (B=1000, seed 12345); paired delta variant - raw, Bonferroni over
(variants x 4 models) for the average column. Saves every val/test score and bootstrap draw to scores_persona_variants.npz;
table persona_variants.json."""
import json, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
import olmo_probe_lib as L
SETS, FAM, B = L.SETS, L.FAM, 1000
KS = [1, 2, 4, 8, 16, 32, 64, 128]; TOPK = list(range(1, 11)); NDRAW = 10
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["llama", "gemma", "qwen", "olmo"]
FZ = np.load("scores_four_models.npz"); CZ = np.load("/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b/transfer/diag/scores_cross_family_tuned.npz")
def raw_scores(m, a, b):
    Z, det = (CZ, "fullprobe") if m in ("llama", "qwen") else (FZ, "raw")
    return Z[[k for k in Z.files if k.startswith(f"{m}__{det}__train_{a}__test_{b}__C")][0]], Z[f"{m}__labels_{b}"]
def single_auc(F, y):
    from sklearn.metrics import roc_auc_score
    return np.array([max(roc_auc_score(y, F[:, j]), 1 - roc_auc_score(y, F[:, j])) for j in range(F.shape[1])])
def variant_cell(key, a, D, P):
    """P: projection spec. ("lin", V) -> X @ V.T ; ("concat", V) -> [X @ V.T, X] ; ("top", T, k) -> top-k traits by training AUC."""
    if P[0] == "top":
        T, k = P[1], P[2]; order = np.argsort(-single_auc(D[a][0] @ T.T, D[a][1])); V = T[order[:k]]; f = lambda X: X @ V.T; extra = {"traits": order[:k].tolist()}
    elif P[0] == "concat": V = P[1]; f = lambda X: np.hstack([X @ V.T, X]); extra = {}
    else: V = P[1]; f = lambda X: X @ V.T; extra = {}
    k_, v = L.cell(key, a, f(D[a][0]), D[a][1], D[a][2], {b: f(D[b][0]) for b in SETS if b != a})
    v.update(extra); return k_, v
res = {"KS": KS, "TOPK": TOPK, "NDRAW": NDRAW, "B": B, "cells": {}, "per_family": {}, "C": {}, "top_traits": {}}; SC = {}
for m in MODELS:
    D = {s: L.LOADERS[m](s) for s in SETS}; T, names = L.trait_matrix(m)
    for s in SETS:
        _, yr = raw_scores(m, "hb" if s != "hb" else "wjb", s); assert (yr == D[s][1]).all(), (m, s, "row order differs from the raw-probe fits")
    Tc = T - T.mean(0); U = np.linalg.svd(Tc, full_matrices=False)[2]   # trait PCs (rows), by explained variance
    rng = np.random.default_rng(0)
    specs = [("PersonaJB", ("lin", T)), ("concat", ("concat", T))] + [(f"PCA_{k}", ("lin", U[:k])) for k in KS + [len(T)] if k <= len(T)] + \
            [(f"top_{k}", ("top", T, k)) for k in TOPK] + [(f"rand_{k}_d{d}", ("lin", T[rng.choice(len(T), k, replace=False)])) for k in TOPK for d in range(NDRAW)]
    OUT = dict(Parallel(n_jobs=8, backend="loky")(delayed(variant_cell)((name, a), a, D, P) for name, P in specs for a in SETS)); print(m, "fits done", flush=True)
    behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {b: i for i, b in enumerate(behs)}
    W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B); bidx = {s: np.array([bi[r["behavior_id"]] for r in D[s][2]]) for s in SETS}
    Tb = {}
    def tab(name, get):
        for a in SETS:
            fams = [f for f in FAM if f != a]; pts, bss = zip(*[L.boot_auc(D[f][1], get(a, f), W[:, bidx[f]]) for f in fams])
            Tb[(name, a)] = (np.mean(pts), np.nanmean(np.stack(bss), 0)); res["per_family"][f"{m}__{name}__{a}"] = dict(zip(fams, map(float, pts)))
        Tb[(name, "avg")] = (np.mean([Tb[(name, a)][0] for a in SETS]), np.mean([Tb[(name, a)][1] for a in SETS], 0))
    tab("raw", lambda a, f: raw_scores(m, a, f)[0])
    for name, _ in specs: tab(name, lambda a, f, name=name: OUT[((name, a), a)]["test"][f])
    for k in TOPK:   # random-k: mean over draws
        for a in SETS + ["avg"]: Tb[(f"rand_{k}", a)] = (np.mean([Tb[(f"rand_{k}_d{d}", a)][0] for d in range(NDRAW)]), np.mean([Tb[(f"rand_{k}_d{d}", a)][1] for d in range(NDRAW)], 0))
    report = ["raw", "PersonaJB", "concat"] + [f"PCA_{k}" for k in KS + [len(T)] if k <= len(T)] + [f"top_{k}" for k in TOPK] + [f"rand_{k}" for k in TOPK]
    nvar = len(report) - 1; zq = 1 - 0.05 / (nvar * 4) / 2
    for name in report:
        for a in SETS + ["avg"]:
            pt, bs = Tb[(name, a)]; d = bs - Tb[("raw", a)][1]
            res["cells"][f"{m}__{name}__{a}"] = {"auc": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                                "delta_vs_raw": float(pt - Tb[("raw", a)][0]), "delta_ci_bonf": [float(np.nanpercentile(d, 100 * (1 - zq))), float(np.nanpercentile(d, 100 * zq))]}
            SC[f"boot__{m}__{name}__{a}"] = bs.astype(np.float32)
    res["C"][m] = {f"{name}__{a}": OUT[((name, a), a)]["C"] for name, _ in specs for a in SETS}
    res["top_traits"][m] = {a: [names[i] for i in OUT[((f"top_{max(TOPK)}", a), a)]["traits"]] for a in SETS}
    for ((name, a), _), v in OUT.items():
        p = f"{m}__{name}__train_{a}"; SC[f"{p}__val_idx"] = v["val_idx"]
        for C, s in v["val_scores"].items(): SC[f"{p}__val_C{C:g}"] = s
        for b, s in v["test"].items(): SC[f"{p}__test_{b}__C{v['C']:g}"] = s
    for s in SETS: SC[f"{m}__labels_{s}"] = D[s][1]; SC[f"{m}__behavior_ids_{s}"] = np.array([r["behavior_id"] for r in D[s][2]])
    print(f"== {m}: avg transfer (delta vs raw, Bonferroni CI)", flush=True)
    for name in report:
        c = res["cells"][f"{m}__{name}__avg"]; print(f"  {name:10s} {c['auc']:.3f}  {c['delta_vs_raw']:+.3f} [{c['delta_ci_bonf'][0]:+.3f},{c['delta_ci_bonf'][1]:+.3f}]", flush=True)
    json.dump(res, open("persona_variants.json", "w"), indent=1); np.savez("scores_persona_variants.npz", **SC)
