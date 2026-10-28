"""Readout selection on the training set's validation split, then transfer. One model per run.
Training set: HB human-jailbreak templates (Gemma: HB + WJB pooled, plain concatenation, as in the paper). Test: every other set.
Detectors: raw activation LR and PersonaJB (LR on projections onto the unit trait vectors of the SAME layer). Both search the
same grid: every saved layer x token position x C (GRID). Per (layer, position) and detector: StandardScaler + LR fitted on a
behaviour-grouped 80/20 split of the training data (HB alone: also template-disjoint; same split for every cell), validation
AUC per C; the (layer, position, C) with the highest validation AUC is refit on all training data and scores the test sets.
Reported: per test set folded AUC and their mean; cluster bootstrap over HarmBench behaviours (B=1000), paired delta
PersonaJB - raw. Saves validation and test scores of every cell to scores_hb_select_{model}.npz, table hb_select_{model}.json.
usage: python hb_select.py MODEL CONFIG_JSON OUTDIR NJOBS [--cv]
--cv: instead of the single 80/20 split, 5-fold CV grouped by HB template (WJB rows, when pooled, by behaviour); a cell's
(layer, position, C) score is the mean validation AUC over folds; output files get the suffix _cv.
CONFIG_JSON: {"acts": "<dir>/{set}_L{layer}.npy" ([n, P, d] float16), "meta": "<dir>/{set}_meta.json" (pair_ids, tail_tokens,
layers), "labels": {set: classified jsonl}, "traits": "<npy [n_traits, n_layers_saved, d] in meta layer order>",
"train": [...], "test": [...]}"""
import json, os, random, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
M, CFG, OUT, NJ = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], int(sys.argv[4]); os.makedirs(OUT, exist_ok=True)
CV = "--cv" in sys.argv; SUF = "_cv" if CV else ""
TRAIN, TEST, SETS = CFG["train"], CFG["test"], CFG["train"] + CFG["test"]
GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; B = 1000
meta = {s: json.load(open(CFG["meta"].format(set=s))) for s in SETS}; LAYERS, TAIL = meta[SETS[0]]["layers"], meta[SETS[0]]["tail_tokens"]
rows, keep = {}, {}
for s in SETS:
    C = {r["pair_id"]: r for r in map(json.loads, open(CFG["labels"][s]))}; ids = meta[s]["pair_ids"]
    keep[s] = np.array([i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")])
    rows[s] = [C[ids[i]] for i in keep[s]]
y = {s: np.array([int(r["jailbroken"]) for r in rows[s]]) for s in SETS}
Rtr = [r for s in TRAIN for r in rows[s]]; ytr = np.concatenate([y[s] for s in TRAIN])
def val_split(R, yy, hb_only):
    for seed in range(300):
        rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in R}); rng.shuffle(beh); vb = set(beh[:int(0.2 * len(beh))])
        if hb_only:
            tp = sorted({r["jailbreak_idx"] for r in R}); rng.shuffle(tp); vt = set(tp[:int(0.2 * len(tp))])
            va = [i for i, r in enumerate(R) if r["behavior_id"] in vb and r["jailbreak_idx"] in vt]; tr = [i for i, r in enumerate(R) if r["behavior_id"] not in vb and r["jailbreak_idx"] not in vt]
        else:
            va = [i for i, r in enumerate(R) if r["behavior_id"] in vb]; tr = [i for i, r in enumerate(R) if r["behavior_id"] not in vb]
        va, tr = np.array(va), np.array(tr)
        if 0 < yy[va].sum() < len(va) and 0 < yy[tr].sum() < len(tr): return tr, va
    raise RuntimeError("no val split")
if CV:
    from sklearn.model_selection import GroupKFold
    grp = np.array([f"t{r['jailbreak_idx']}" if r["attack_type"] == "human_jailbreak" else f"b{r['behavior_id']}" for r in Rtr])
    FOLDS = list(GroupKFold(5).split(np.zeros(len(ytr)), ytr, grp))
else:
    FOLDS = [val_split(Rtr, ytr, TRAIN == ["hb"])]
TR, VA = FOLDS[0]
fold = lambda yy, s: max(roc_auc_score(yy, s), 1 - roc_auc_score(yy, s))
TV = np.load(CFG["traits"], mmap_mode="r")   # [n_traits, n_layers, d]
def fit(C, X, yy):
    sc = StandardScaler().fit(X); m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), yy); return lambda Z: m.decision_function(sc.transform(Z))
def cell(li, p, det):
    T = np.asarray(TV[:, li], np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    f = (lambda X: X) if det == "raw" else (lambda X: X @ T.T)
    X = {s: f(np.load(CFG["acts"].format(set=s, layer=LAYERS[li]), mmap_mode="r")[keep[s], p].astype(np.float64)) for s in SETS}
    Xtr = np.vstack([X[s] for s in TRAIN])
    vs, vauc = {}, {}
    for C in GRID:   # out-of-fold validation scores; AUC averaged over folds
        oof = np.full(len(ytr), np.nan, np.float32); a = []
        for tr, va in FOLDS: oof[va] = fit(C, Xtr[tr], ytr[tr])(Xtr[va]); a.append(fold(ytr[va], oof[va]))
        vs[C] = oof; vauc[C] = float(np.mean(a))
    Cb = max(GRID, key=lambda c: vauc[c]); sc = fit(Cb, Xtr, ytr)
    return (LAYERS[li], p, det), {"C": Cb, "val_auc": vauc, "val_scores": vs, "test": {s: sc(X[s]).astype(np.float32) for s in TEST}}
R_ = dict(Parallel(n_jobs=NJ, backend="loky", verbose=2)(delayed(cell)(li, p, det) for li in range(len(LAYERS)) for p in range(len(TAIL)) for det in ("PersonaJB", "raw")))
behs = sorted({r["behavior_id"] for s in SETS for r in rows[s]}); bi = {b: i for i, b in enumerate(behs)}
W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)
def boot(yy, s, bo):
    sg = 1 if roc_auc_score(yy, s) >= 0.5 else -1
    return roc_auc_score(yy, sg * s), np.array([roc_auc_score(yy, sg * s, sample_weight=W[k][bo]) if (W[k][bo] * yy).sum() > 0 and (W[k][bo] * (1 - yy)).sum() > 0 else np.nan for k in range(B)])
res = {"model": M, "cv": CV, "train": TRAIN, "test": TEST, "layers": LAYERS, "tail_tokens": TAIL, "n_val": [int(len(v)) for _, v in FOLDS], "val_pos": [int(ytr[v].sum()) for _, v in FOLDS], "grid": {}, "selected": {}}; SC = {}; Tb = {}
for (l, p, det), v in R_.items():
    res["grid"][f"{det}__L{l}__p{p}"] = {"val_auc": max(v["val_auc"].values()), "C": v["C"], "test_auc": {s: float(fold(y[s], v["test"][s])) for s in TEST}}
    q = f"{det}__L{l}__p{p}"
    for C, s in v["val_scores"].items(): SC[f"{q}__val_C{C:g}"] = s
    for s, sc in v["test"].items(): SC[f"{q}__test_{s}__C{v['C']:g}"] = sc
for det in ("PersonaJB", "raw"):
    (l, p, _), v = max(((k, v) for k, v in R_.items() if k[2] == det), key=lambda kv: max(kv[1]["val_auc"].values()))
    pts, bss = zip(*[boot(y[s], v["test"][s], np.array([bi[r["behavior_id"]] for r in rows[s]])) for s in TEST])
    Tb[det] = (np.mean(pts), np.nanmean(np.stack(bss), 0))
    res["selected"][det] = {"layer": l, "position": p, "token": TAIL[p], "C": v["C"], "val_auc": max(v["val_auc"].values()), "test_auc": dict(zip(TEST, map(float, pts))),
                            "avg": float(Tb[det][0]), "avg_ci95": [float(np.nanpercentile(Tb[det][1], 2.5)), float(np.nanpercentile(Tb[det][1], 97.5))]}
    SC[f"boot__{det}"] = Tb[det][1].astype(np.float32)
d = Tb["PersonaJB"][1] - Tb["raw"][1]; res["delta"] = {"delta": float(Tb["PersonaJB"][0] - Tb["raw"][0]), "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
for k, (_, v) in enumerate(FOLDS): SC[f"val_idx_fold{k}"] = v
for s in SETS: SC[f"labels_{s}"] = y[s]; SC[f"pair_ids_{s}"] = np.array([str(r["pair_id"]) for r in rows[s]]); SC[f"behavior_ids_{s}"] = np.array([r["behavior_id"] for r in rows[s]])
json.dump(res, open(f"{OUT}/hb_select_{M}{SUF}.json", "w"), indent=1); np.savez(f"{OUT}/scores_hb_select_{M}{SUF}.npz", **SC)
for det in ("PersonaJB", "raw"):
    s = res["selected"][det]; print(f"{M}{SUF} {det:9s} L{s['layer']} {s['token']!r} C={s['C']:g} val {s['val_auc']:.3f} | " + " ".join(f"{t}:{a:.3f}" for t, a in s["test_auc"].items()) + f" | avg {s['avg']:.3f} [{s['avg_ci95'][0]:.3f},{s['avg_ci95'][1]:.3f}]")
print(f"{M}{SUF} delta PersonaJB-raw {res['delta']['delta']:+.3f} [{res['delta']['ci95'][0]:+.3f},{res['delta']['ci95'][1]:+.3f}]")
