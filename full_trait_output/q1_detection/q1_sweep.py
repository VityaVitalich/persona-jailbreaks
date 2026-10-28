"""Q1: can persona directions detect jailbreaks, and what does each detector lose when the persona component is removed?
Gemma-4-31B-it. Training set HB human-jailbreak + WJB pooled (as in the paper), test = every transfer family. Readout
selection exactly as hb_select.py --cv: 5-fold CV on the training set grouped by HB template (WJB rows by behaviour);
every detector picks its own (layer, position, C) by mean validation folded AUC, is refit on all training rows and
scores the transfer sets. Folded AUC max(a, 1-a) everywhere.
Detectors, at every saved layer x tail position:
  LR on features (StandardScaler + LogisticRegression, C grid):
    raw          the activation itself (linear probe)
    PersonaJB    projections onto the 240 unit trait vectors of the same layer
    PCA          top-240 PCs of the training-fold activations
    gauss_d{k}   projections onto 240 random Gaussian unit directions (3 draws)
    alpaca_pca   top-240 PCs of 797 Alpaca instruction activations (same layer/position, chat template)
    wiki_pca     top-240 PCs of 2000 Wikipedia passage activations (wrapped as a user turn, same layer/position)
  single direction, score = h . v (no fitted hyperparameter):
    jb_dir       mean(jailbroken) - mean(refused) on the training fold
    refusal      Arditi harmful - harmless mean difference (gemma_refusal_dir, same layer/position)
    alpaca_mean  mean Alpaca activation
    wiki_mean    mean Wikipedia activation
Conditions: full; no_persona = every activation (detection sets AND the Alpaca/Wikipedia/refusal inputs the detectors
are built from) has its component in span(240 trait vectors of that layer) projected out; no_random = same with a
random 240-dim Gaussian span (null for "remove any 240 dims"); only_persona = keep only the component in the persona
span. PersonaJB is undefined under no_persona and identical to full under only_persona (not rerun).
Saves: every cell's out-of-fold validation scores per C and transfer scores at the chosen C (scores_q1_{MODEL}.npz),
per-cell validation AUCs, selections, bootstrap (HarmBench-behaviour cluster, B=1000) for avg transfer and the paired
deltas {no_persona, no_random, only_persona} - full at (a) the full detector's (layer, position) with C re-tuned and (b) a
fully re-selected readout (q1_{MODEL}.json).
usage: python q1_sweep.py MODEL CONFIG_JSON OUTDIR NJOBS [--smoke]
CONFIG_JSON = hb_select config + "ctrl_acts": "<dir>/{set}_L{layer}.npy" (sets alpaca, wiki), "refusal": npy [7, 5, d]"""
import json, os, random, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
M, CFG, OUT, NJ = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], int(sys.argv[4]); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
TRAIN, TEST, SETS = CFG["train"], CFG["test"], CFG["train"] + CFG["test"]
GRID = [1e-2] if SMOKE else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; B = 200 if SMOKE else 1000; K = 240; NG = 3
CONDS = ("full", "no_persona", "no_random", "only_persona")
LR_DETS = ["raw", "PersonaJB", "PCA"] + [f"gauss_d{k}" for k in range(NG)] + ["alpaca_pca", "wiki_pca"]
DIR_DETS = ["jb_dir", "refusal", "alpaca_mean", "wiki_mean"]
meta = {s: json.load(open(CFG["meta"].format(set=s))) for s in SETS}; LAYERS, TAIL = meta[SETS[0]]["layers"], meta[SETS[0]]["tail_tokens"]
rows, keep = {}, {}
for s in SETS:
    C = {r["pair_id"]: r for r in map(json.loads, open(CFG["labels"][s]))}; ids = meta[s]["pair_ids"]
    keep[s] = np.array([i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")])
    rows[s] = [C[ids[i]] for i in keep[s]]
y = {s: np.array([int(r["jailbroken"]) for r in rows[s]]) for s in SETS}
Rtr = [r for s in TRAIN for r in rows[s]]; ytr = np.concatenate([y[s] for s in TRAIN])
grp = np.array([f"t{r['jailbreak_idx']}" if r["attack_type"] == "human_jailbreak" else f"b{r['behavior_id']}" for r in Rtr])
FOLDS = list(GroupKFold(5).split(np.zeros(len(ytr)), ytr, grp))
fold = lambda yy, s: max(roc_auc_score(yy, s), 1 - roc_auc_score(yy, s))
TV = np.load(CFG["traits"], mmap_mode="r")   # [n_traits, n_layers, d]
REF = np.load(CFG["refusal"], mmap_mode="r")  # [n_layers, n_pos, d]
unit = lambda M: M / np.linalg.norm(M, axis=-1, keepdims=True)
orth = lambda V: np.linalg.qr(np.asarray(V, np.float64).T)[0]   # [d, k] orthonormal basis of the row span
def pcs(X, k=K):
    X = X - X.mean(0); return np.linalg.svd(X, full_matrices=False)[2][:k]
def lr_det(make, Xtr, Xte):
    """make(Xfit) -> feature map. Out-of-fold validation per C (feature map fitted on the training fold), refit on all."""
    vs = {C: np.full(len(ytr), np.nan, np.float32) for C in GRID}; a = {C: [] for C in GRID}
    for tr, va in FOLDS:
        f = make(Xtr[tr]); Ftr, Fva = f(Xtr[tr]), f(Xtr[va]); sc = StandardScaler().fit(Ftr); Ftr, Fva = sc.transform(Ftr), sc.transform(Fva)
        for C in GRID:
            vs[C][va] = LogisticRegression(C=C, max_iter=5000).fit(Ftr, ytr[tr]).decision_function(Fva); a[C].append(fold(ytr[va], vs[C][va]))
    vauc = {C: float(np.mean(a[C])) for C in GRID}; Cb = max(GRID, key=lambda c: vauc[c])
    f = make(Xtr); F = f(Xtr); sc = StandardScaler().fit(F); m = LogisticRegression(C=Cb, max_iter=5000).fit(sc.transform(F), ytr)
    return {"C": Cb, "val_auc": vauc, "val_scores": vs, "test": {s: m.decision_function(sc.transform(f(X))).astype(np.float32) for s, X in Xte.items()}}
def dir_det(vfun, Xtr, Xte):
    """vfun(train-row indices) -> direction; validation = out-of-fold projections."""
    oof = np.full(len(ytr), np.nan, np.float32); a = []
    for tr, va in FOLDS: oof[va] = Xtr[va] @ vfun(tr); a.append(fold(ytr[va], oof[va]))
    v = vfun(np.arange(len(ytr)))
    return {"C": None, "val_auc": {None: float(np.mean(a))}, "val_scores": {None: oof}, "test": {s: (X @ v).astype(np.float32) for s, X in Xte.items()}}
GAUSS = [unit(np.random.default_rng(k).standard_normal((K, TV.shape[2]))) for k in range(NG)]
Q_RAND = orth(np.random.default_rng(999).standard_normal((K, TV.shape[2])))
def task(li, p, cond):
    L = LAYERS[li]; T = unit(np.asarray(TV[:, li], np.float64))
    X = {s: np.load(CFG["acts"].format(set=s, layer=L), mmap_mode="r")[keep[s], p].astype(np.float64) for s in SETS}
    A = {c: np.load(CFG["ctrl_acts"].format(set=c, layer=L), mmap_mode="r")[:, p].astype(np.float64) for c in ("alpaca", "wiki")}
    r = np.asarray(REF[li, p], np.float64)
    Qp = orth(T); info = {}
    share = lambda V, Q: float(np.mean(np.sum((np.atleast_2d(V) @ Q) ** 2, 1) / np.sum(np.atleast_2d(V) ** 2, 1)))
    Xtr0 = np.vstack([X[s] for s in TRAIN])
    info["persona_share_train_acts"] = share(Xtr0, Qp); info["random_share_train_acts"] = share(Xtr0, Q_RAND)
    if cond != "full":
        Q = Q_RAND if cond == "no_random" else Qp
        rm = (lambda M: (M @ Q) @ Q.T) if cond == "only_persona" else (lambda M: M - (M @ Q) @ Q.T)
        X = {s: rm(M) for s, M in X.items()}; A = {c: rm(M) for c, M in A.items()}; r = rm(r)
    Xtr = np.vstack([X[s] for s in TRAIN]); Xte = {s: X[s] for s in TEST}; del X, Xtr0
    static = lambda V: (lambda _: (lambda Z: Z @ V.T))
    makers = {"raw": lambda _: (lambda Z: Z), "PCA": lambda Xf: (lambda V: lambda Z: Z @ V.T)(pcs(Xf)),
              "alpaca_pca": static(pcs(A["alpaca"])), "wiki_pca": static(pcs(A["wiki"]))}
    if cond in ("full", "no_random"): makers["PersonaJB"] = static(T)
    for k in range(NG): makers[f"gauss_d{k}"] = static(GAUSS[k])
    out = {}
    for det in LR_DETS:
        if det in makers: out[det] = lr_det(makers[det], Xtr, Xte)
    vfix = {"refusal": unit(r), "alpaca_mean": unit(A["alpaca"].mean(0)), "wiki_mean": unit(A["wiki"].mean(0))}
    out["jb_dir"] = dir_det(lambda idx: unit(Xtr[idx][ytr[idx] == 1].mean(0) - Xtr[idx][ytr[idx] == 0].mean(0)), Xtr, Xte)
    for det, v in vfix.items(): out[det] = dir_det(lambda idx, v=v: v, Xtr, Xte)
    if cond == "full":   # how much of each detector's direction / basis lies in the persona span
        dmu = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        info["persona_share"] = {"jb_dir": share(dmu, Qp), **{d: share(v, Qp) for d, v in vfix.items()},
                                 "PCA": share(pcs(Xtr), Qp), "alpaca_pca": share(pcs(A["alpaca"]), Qp), "wiki_pca": share(pcs(A["wiki"]), Qp),
                                 "gauss": float(np.mean([share(G, Qp) for G in GAUSS]))}
    print(f"done L{L} p{p} {cond}: " + " ".join(f"{d}:{max(v['val_auc'].values()):.3f}" for d, v in out.items()), flush=True)
    return (L, p, cond), out, info
cells = [(li, p) for li in range(len(LAYERS)) for p in range(len(TAIL))]
if SMOKE: cells = [(LAYERS.index(30), 4)]
R_ = Parallel(n_jobs=NJ, backend="loky", verbose=0)(delayed(task)(li, p, c) for li, p in cells for c in CONDS)
behs = sorted({r["behavior_id"] for s in SETS for r in rows[s]}); bi = {b: i for i, b in enumerate(behs)}
W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)
BO = {s: np.array([bi[r["behavior_id"]] for r in rows[s]]) for s in TEST}
def boot_avg(test):
    """point avg folded transfer AUC + B bootstrap reps (sign fixed per set by the point estimate)."""
    pts, reps = [], []
    for s in TEST:
        yy, sc, bo = y[s], test[s], BO[s]; sg = 1 if roc_auc_score(yy, sc) >= 0.5 else -1; pts.append(roc_auc_score(yy, sg * sc))
        reps.append([roc_auc_score(yy, sg * sc, sample_weight=W[k][bo]) if (W[k][bo] * yy).sum() > 0 and (W[k][bo] * (1 - yy)).sum() > 0 else np.nan for k in range(B)])
    return pts, np.nanmean(np.array(reps), 0)
G_ = {}; info = {}; SC = {}
for (L, p, cond), out, inf in R_:
    info[f"L{L}__p{p}__{cond}"] = inf
    for det, v in out.items():
        G_[(det, cond, L, p)] = v; q = f"{det}__{cond}__L{L}__p{p}"
        for C, s in v["val_scores"].items(): SC[f"{q}__val_C{C if C is None else f'{C:g}'}"] = s
        for s, sc in v["test"].items(): SC[f"{q}__test_{s}"] = sc
vbest = lambda v: max(v["val_auc"].values())
def select(det, cond, L=None, p=None):
    c = [(k, v) for k, v in G_.items() if k[0] == det and k[1] == cond and (L is None or (k[2], k[3]) == (L, p))]
    return max(c, key=lambda kv: vbest(kv[1])) if c else None
res = {"model": M, "train": TRAIN, "test": TEST, "layers": LAYERS, "tail_tokens": TAIL, "grid": GRID,
       "n_train": int(len(ytr)), "n_pos_train": int(ytr.sum()), "n_val": [int(len(v)) for _, v in FOLDS], "val_pos": [int(ytr[v].sum()) for _, v in FOLDS],
       "n_test": {s: [int(len(y[s])), int(y[s].sum())] for s in TEST}, "info": info,
       "cells": {f"{d}__{c}__L{L}__p{p}": {"val_auc": vbest(v), "C": v["C"], "test_auc": {s: float(fold(y[s], v["test"][s])) for s in TEST}} for (d, c, L, p), v in G_.items()},
       "selected": {}, "delta": {}}
BOOT = {}
for det in LR_DETS + DIR_DETS:
    for cond in CONDS:
        for mode in ("reselect", "fixed"):
            if mode == "fixed" and cond == "full": continue
            if mode == "fixed":
                (_, _, L0, p0), _ = select(det, "full"); s = select(det, cond, L0, p0)
            else: s = select(det, cond)
            if s is None: continue
            (_, _, L, p), v = s; pts, reps = boot_avg(v["test"]); BOOT[(det, cond, mode)] = (float(np.mean(pts)), reps)
            res["selected"][f"{det}__{cond}__{mode}"] = {"layer": L, "position": p, "token": TAIL[p], "C": v["C"], "val_auc": vbest(v),
                "test_auc": dict(zip(TEST, map(float, pts))), "avg": float(np.mean(pts)), "avg_ci95": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]}
            SC[f"boot__{det}__{cond}__{mode}"] = reps.astype(np.float32)
def agg(det, cond, mode):   # Gaussian draws -> their mean
    ks = [f"gauss_d{k}" for k in range(NG)] if det == "gauss" else [det]
    b = [BOOT[(k, cond, mode)] for k in ks if (k, cond, mode) in BOOT]
    return (float(np.mean([x[0] for x in b])), np.mean([x[1] for x in b], 0)) if b else None
for det in ["raw", "PersonaJB", "PCA", "gauss", "alpaca_pca", "wiki_pca"] + DIR_DETS:
    base = agg(det, "full", "reselect")
    if base is None: continue
    res["selected"][f"{det}__full__summary"] = {"avg": base[0], "avg_ci95": [float(np.nanpercentile(base[1], 2.5)), float(np.nanpercentile(base[1], 97.5))]}
    for cond in ("no_persona", "no_random", "only_persona"):
        for mode in ("fixed", "reselect"):
            o = agg(det, cond, mode)
            if o is None: continue
            d = o[1] - base[1]
            res["delta"][f"{det}__{cond}__{mode}"] = {"avg": o[0], "delta": o[0] - base[0], "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
for k, (_, v) in enumerate(FOLDS): SC[f"val_idx_fold{k}"] = v
SC["labels_train"] = ytr; SC["pair_ids_train"] = np.array([f"{s}:{r['pair_id']}" for s in TRAIN for r in rows[s]])
for s in TEST: SC[f"labels_{s}"] = y[s]; SC[f"pair_ids_{s}"] = np.array([str(r["pair_id"]) for r in rows[s]]); SC[f"behavior_ids_{s}"] = np.array([r["behavior_id"] for r in rows[s]])
suf = "_smoke" if SMOKE else ""
json.dump(res, open(f"{OUT}/q1_{M}{suf}.json", "w"), indent=1); np.savez(f"{OUT}/scores_q1_{M}{suf}.npz", **SC)
for k, v in res["selected"].items():
    if k.endswith("__full__reselect"): print(f"{k:32s} L{v['layer']} p{v['position']} C={v['C']} val {v['val_auc']:.3f} avg {v['avg']:.3f} [{v['avg_ci95'][0]:.3f},{v['avg_ci95'][1]:.3f}]")
for k, v in res["delta"].items(): print(f"{k:36s} {v['avg']:.3f} delta {v['delta']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
