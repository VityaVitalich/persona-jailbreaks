"""Mixed training (LOFO) for OLMo-3-7B and Gemma-4-31B: PersonaJB LR vs raw-activation LR, as in
qwen3_32b/probe_methods/probe_methods.py: train on all sets but the held-out family (each training set equal total weight),
C tuned on a behaviour-grouped validation split of the training data, refit, score the held-out family.
PersonaJB: OLMo 240 unfiltered trait vectors (row 16), Gemma 240 paper vectors (layer 30). Same readouts as four_models.py.
Cluster bootstrap over HarmBench behaviours (B=1000, seed 12345), paired delta PersonaJB - raw, Bonferroni over
(5 families + avg) x 2 models. Saves val/test scores and bootstrap draws to scores_lofo_pjb_raw.npz; table lofo_pjb_raw.json."""
import glob, json, os, numpy as np, torch
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import olmo_probe_lib as L
from olmo_probe_lib import gemma, olmo, val_split, fold, GRID
SETS, FAM, B = L.SETS, L.FAM, 1000
O = "/Users/viktor/persona-vector-jbing/full_trait_output/olmo3_7b"
def tv(pat, row):
    P = sorted(glob.glob(pat)); T = np.stack([torch.load(p, map_location="cpu", weights_only=False)["vector"][row].float().numpy() for p in P]).astype(np.float64)
    return T / np.linalg.norm(T, axis=1, keepdims=True)
def fit(C, X, y, w):
    sc = StandardScaler().fit(X); m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y, sample_weight=w)
    return lambda Z: m.decision_function(sc.transform(Z))
def cell(m, det, f, D, V):
    trs = [s for s in SETS if s != f]; F = lambda X: X if V is None else X @ V.T
    X = np.vstack([F(D[s][0]) for s in trs]); y = np.concatenate([D[s][1] for s in trs]); R = [r for s in trs for r in D[s][2]]
    w = np.concatenate([np.full(len(D[s][1]), len(y) / len(trs) / len(D[s][1])) for s in trs])
    tr, va = val_split(R, y, False); vs = {C: fit(C, X[tr], y[tr], w[tr])(X[va]).astype(np.float32) for C in GRID}
    Cb = max(GRID, key=lambda c: fold(y[va], vs[c]))
    return (m, det, f), {"C": Cb, "val_idx": va, "val_scores": vs, "test": fit(Cb, X, y, w)(F(D[f][0])).astype(np.float32)}
res, SC = {"B": B, "cells": {}}, {}
NB = (len(FAM) + 1) * 2; zq = 1 - 0.05 / NB / 2
for m, loader, V in (("olmo", olmo, tv(f"{O}/trait_vectors/all_traits_no_filter/*.pt", 16)), ("gemma", gemma, tv(f"{O}/gemma_data/trait_vectors/*.pt", 0))):
    D = {s: loader(s) for s in SETS}
    OUT = dict(Parallel(n_jobs=8, backend="loky")(delayed(cell)(m, det, f, D, VV) for det, VV in (("PersonaJB", V), ("raw", None)) for f in FAM)); print(m, "fits done", flush=True)
    behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {b: i for i, b in enumerate(behs)}
    W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)
    T = {}
    for (_, det, f), v in OUT.items():
        y = D[f][1]; s = v["test"]; bo = np.array([bi[r["behavior_id"]] for r in D[f][2]]); sg = 1 if roc_auc_score(y, s) >= 0.5 else -1
        T[(det, f)] = (roc_auc_score(y, sg * s), np.array([roc_auc_score(y, sg * s, sample_weight=W[k][bo]) if (W[k][bo] * y).sum() > 0 and (W[k][bo] * (1 - y)).sum() > 0 else np.nan for k in range(B)]))
        p = f"{m}__{det}__lofo_{f}"; SC[f"{p}__val_idx"] = v["val_idx"]; SC[f"{p}__test__C{v['C']:g}"] = s
        for C, vs in v["val_scores"].items(): SC[f"{p}__val_C{C:g}"] = vs
    for det in ("PersonaJB", "raw"): T[(det, "avg")] = (np.mean([T[(det, f)][0] for f in FAM]), np.mean([T[(det, f)][1] for f in FAM], 0))
    for f in FAM + ["avg"]:
        d = T[("PersonaJB", f)][1] - T[("raw", f)][1]
        res["cells"][f"{m}__{f}"] = {"PersonaJB": float(T[("PersonaJB", f)][0]), "raw": float(T[("raw", f)][0]), "delta": float(T[("PersonaJB", f)][0] - T[("raw", f)][0]),
                                     "delta_ci_bonf": [float(np.nanpercentile(d, 100 * (1 - zq))), float(np.nanpercentile(d, 100 * zq))],
                                     "C": None if f == "avg" else {det: OUT[(m, det, f)]["C"] for det in ("PersonaJB", "raw")}}
        for det in ("PersonaJB", "raw"): SC[f"boot__{m}__{det}__{f}"] = T[(det, f)][1].astype(np.float32)
    for s in SETS: SC[f"{m}__labels_{s}"] = D[s][1]; SC[f"{m}__behavior_ids_{s}"] = np.array([r["behavior_id"] for r in D[s][2]])
json.dump(res, open("lofo_pjb_raw.json", "w"), indent=1); np.savez("scores_lofo_pjb_raw.npz", **SC)
for m in ("olmo", "gemma"):
    print(m, " ".join(f"{f}: PJB {res['cells'][f'{m}__{f}']['PersonaJB']:.3f} raw {res['cells'][f'{m}__{f}']['raw']:.3f} d {res['cells'][f'{m}__{f}']['delta']:+.3f}[{res['cells'][f'{m}__{f}']['delta_ci_bonf'][0]:+.3f},{res['cells'][f'{m}__{f}']['delta_ci_bonf'][1]:+.3f}]" for f in FAM + ["avg"]))
