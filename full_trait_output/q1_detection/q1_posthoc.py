"""Q1 option A: remove the persona component AFTER training. For every detector, take its readout selected by q1_sweep.py
(full condition, layer/position/C from q1_{MODEL}.json), refit on all full-activation training rows exactly as the sweep's
final fit, then score the transfer sets with the test activations (a) unchanged (must reproduce the sweep's saved
scores), (b) with the persona span projected out (h - QQ^T h), (c) with the same random K-dim span as the sweep
projected out, (d) with only the persona component kept. Training data and the detector are never changed.
For fixed directions (refusal, Alpaca/Wiki means) and the JB direction this equals removal before training, so they
are included as a consistency check against the sweep's deltas.
Also reports the share of each detector's effective raw-space weight in the persona span (raw: w / sigma; feature
detectors: V^T (w / sigma)); a random K-dim span holds ~K/d.
Folded AUC max(a, 1-a) per condition and set (as everywhere in the project), cluster
bootstrap over HarmBench behaviours (same seed/B as the sweep), paired delta vs (a).
usage: python q1_posthoc.py TAG CONFIG_JSON SWEEP_JSON OUTDIR   (TAG = {model}_{hb|hbwjb})"""
import json, os, scipy.linalg, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
M, CFG, SW, OUT = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4]; os.makedirs(OUT, exist_ok=True)
TRAIN, TEST, SETS = CFG["train"], CFG["test"], CFG["train"] + CFG["test"]; NG, B = 3, 1000; AVG = [s for s in TEST if s != "wjb"]
meta = {s: json.load(open(CFG["meta"].format(set=s))) for s in SETS}; LAYERS = meta[SETS[0]]["layers"]
rows, keep = {}, {}
for s in SETS:
    C = {r["pair_id"]: r for r in map(json.loads, open(CFG["labels"][s]))}; ids = meta[s]["pair_ids"]
    keep[s] = np.array([i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")])
    rows[s] = [C[ids[i]] for i in keep[s]]
y = {s: np.array([int(r["jailbroken"]) for r in rows[s]]) for s in SETS}; ytr = np.concatenate([y[s] for s in TRAIN])
TV = np.load(CFG["traits"], mmap_mode="r"); K = TV.shape[0]
unit = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)
orth = lambda V: np.linalg.qr(np.asarray(V, np.float64).T)[0]
def pcs(X, k=K):   # gesdd can fail to converge on Qwen's massive-activation matrices -> fall back to gesvd
    X = X - X.mean(0)
    try: return np.linalg.svd(X, full_matrices=False)[2][:k]
    except np.linalg.LinAlgError: return scipy.linalg.svd(X, full_matrices=False, lapack_driver="gesvd")[2][:k]
GAUSS = [unit(np.random.default_rng(k).standard_normal((K, TV.shape[2]))) for k in range(NG)]
Q_RAND = orth(np.random.default_rng(999).standard_normal((K, TV.shape[2])))   # same null span as q1_sweep.py
DETS = ["raw", "PersonaJB", "PCA"] + [f"gauss_d{k}" for k in range(NG)] + ["alpaca_pca", "wiki_pca", "jb_dir", "refusal", "alpaca_mean", "wiki_mean"]
SCORES = {"pair_ids_" + s: np.array([str(r["pair_id"]) for r in rows[s]]) for s in TEST}; res = {"model": M, "test": TEST, "detectors": {}}
cache = {}
def load(L, p):
    if (L, p) not in cache:
        cache.clear(); li = LAYERS.index(L)
        X = {s: np.load(CFG["acts"].format(set=s, layer=L), mmap_mode="r")[keep[s], p].astype(np.float64) for s in SETS}
        A = {c: np.load(CFG["ctrl_acts"].format(set=c, layer=L), mmap_mode="r")[:, p].astype(np.float64) for c in ("alpaca", "wiki", "harmful")}
        cache[(L, p)] = (X, A, A["harmful"].mean(0) - A["alpaca"].mean(0), unit(np.asarray(TV[:, li], np.float64)))
    return cache[(L, p)]
for det in DETS:
    sel = SW["selected"][f"{det}__full__reselect"]; L, p, Cb = sel["layer"], sel["position"], sel["C"]
    X, A, r, T = load(L, p); Xtr = np.vstack([X[s] for s in TRAIN]); Qp = orth(T)
    V = {"raw": None, "PersonaJB": T, "PCA": pcs(Xtr) if det == "PCA" else None, "alpaca_pca": pcs(A["alpaca"]) if det == "alpaca_pca" else None,
         "wiki_pca": pcs(A["wiki"]) if det == "wiki_pca" else None, **{f"gauss_d{k}": GAUSS[k] for k in range(NG)}}.get(det)
    if det in ("jb_dir", "refusal", "alpaca_mean", "wiki_mean"):
        v = {"jb_dir": lambda: Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0), "refusal": lambda: r,
             "alpaca_mean": lambda: A["alpaca"].mean(0), "wiki_mean": lambda: A["wiki"].mean(0)}[det]()
        w_eff = unit(v); score = lambda Z: Z @ w_eff
    else:
        f = (lambda Z: Z) if V is None else (lambda Z, V=V: Z @ V.T)
        sc = StandardScaler().fit(f(Xtr)); m = LogisticRegression(C=Cb, max_iter=5000).fit(sc.transform(f(Xtr)), ytr)
        score = lambda Z: m.decision_function(sc.transform(f(Z)))
        ws = m.coef_[0] / sc.scale_; w_eff = ws if V is None else V.T @ ws
    share = lambda Q: float(np.sum((w_eff @ Q) ** 2) / np.sum(w_eff ** 2))
    conds = {"full": lambda Z: Z, "no_persona": lambda Z: Z - (Z @ Qp) @ Qp.T, "no_random": lambda Z: Z - (Z @ Q_RAND) @ Q_RAND.T, "only_persona": lambda Z: (Z @ Qp) @ Qp.T}
    S = {c: {s: score(t(X[s])).astype(np.float32) for s in TEST} for c, t in conds.items()}
    saved = {s: float(max(roc_auc_score(y[s], S["full"][s]), 1 - roc_auc_score(y[s], S["full"][s]))) for s in TEST}
    assert all(abs(saved[s] - sel["test_auc"][s]) < 2e-3 for s in TEST), (det, saved, sel["test_auc"])   # refit reproduces the sweep
    res["detectors"][det] = {"layer": L, "position": p, "C": Cb, "w_share_persona": share(Qp), "w_share_random": share(Q_RAND), "conds": S}
    for c in conds:
        for s in TEST: SCORES[f"{det}__{c}__test_{s}"] = S[c][s]
    print(f"{det:12s} L{L} p{p} C={Cb}  w in persona {share(Qp):.3f} (random {share(Q_RAND):.3f})", flush=True)
behs = sorted({r["behavior_id"] for s in TEST for r in rows[s]}); bi = {b: i for i, b in enumerate(behs)}
W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)
BO = {s: np.array([bi[r["behavior_id"]] for r in rows[s]]) for s in AVG}
def auc_reps(yy, sc, bo, sg):
    return roc_auc_score(yy, sg * sc), np.array([roc_auc_score(yy, sg * sc, sample_weight=W[k][bo]) if (W[k][bo] * yy).sum() > 0 and (W[k][bo] * (1 - yy)).sum() > 0 else np.nan for k in range(B)])
out = {"model": M, "test": TEST, "avg_sets": AVG, "note": "folded AUC per condition and set; delta = cond - full, paired bootstrap", "detectors": {}}
for det, d in res["detectors"].items():
    o = {k: d[k] for k in ("layer", "position", "C", "w_share_persona", "w_share_random")}
    rep = {}
    for c, S in d["conds"].items():
        pr = [auc_reps(y[s], S[s], BO[s], 1 if roc_auc_score(y[s], S[s]) >= 0.5 else -1) for s in AVG]; rep[c] = np.nanmean([x[1] for x in pr], 0)
        o[c] = {"test_auc": {s: float(max(roc_auc_score(y[s], S[s]), 1 - roc_auc_score(y[s], S[s]))) for s in TEST}, "avg": float(np.mean([x[0] for x in pr])),
                "avg_ci95": [float(np.nanpercentile(rep[c], 2.5)), float(np.nanpercentile(rep[c], 97.5))]}
        if c != "full":
            dd = rep[c] - rep["full"]; o[c]["delta"] = o[c]["avg"] - o["full"]["avg"]; o[c]["delta_ci95"] = [float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))]
        SCORES[f"boot__{det}__{c}"] = rep[c].astype(np.float32)
    out["detectors"][det] = o
for s in TEST: SCORES[f"labels_{s}"] = y[s]
json.dump(out, open(f"{OUT}/q1_posthoc_{M}.json", "w"), indent=1); np.savez(f"{OUT}/scores_q1_posthoc_{M}.npz", **SCORES)
for det, o in out["detectors"].items():
    print(f"{det:12s} full {o['full']['avg']:.3f} | " + " | ".join(f"{c} {o[c]['delta']:+.3f} [{o[c]['delta_ci95'][0]:+.3f},{o[c]['delta_ci95'][1]:+.3f}]" for c in ("no_persona", "no_random", "only_persona")))
