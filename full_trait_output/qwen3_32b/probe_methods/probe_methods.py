#!/usr/bin/env python3
"""Can the probe be improved on Qwen? Methods x training regimes, Qwen3-32B (L48, last prompt token) and Llama-3.1-8B
(L16, control). All hyperparameters tuned on a behaviour-grouped validation split of the TRAINING data (HB alone:
behaviour- and template-disjoint), then refit on all training data.
Methods: LR on PersonaJB projections / Gaussian projections (3 draws) / raw activation (baselines); PCA->LR (k, C);
difference of means (raw, and on standardized features); concat [PersonaJB projections, raw] -> LR; gradient boosting
(HistGB) on PersonaJB projections and on PCA-128; 1-hidden-layer MLP on PersonaJB projections and on raw.
Regimes: single (train on one of HB/WJB/PAP/GCG/PAIR/GPTFuzz/PEZ, test on the transfer families minus itself) and
LOFO (train on all sets but one family, each set weighted equally, test on the held-out family).
CIs: behaviour-clustered bootstrap (B=1000) shared across test sets; paired deltas vs raw LR, Bonferroni.
usage: probe_methods.py QWEN_SWEEP_DIR QWEN_TRANSFER_DIR LLAMA_DIR OUTDIR NJOBS"""
import json, os, random, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
QS, QT, LD, OUT, NJ = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]); os.makedirs(OUT, exist_ok=True)
from joblib import Parallel, delayed
SETS = ["hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"]; FAM = ["gcg", "pair", "pap", "gptfuzz", "pez"]
CG = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; KS = [8, 16, 32, 64, 128, 256, 512]; B = 1000

def load_qwen():
    import torch
    D = {}
    for s in SETS:
        cf = f"{QS}/in/classified_{s}.jsonl" if s in ("hb", "wjb", "pap") else f"{QT}/classified_{s}.jsonl"
        C = {r["pair_id"]: r for r in map(json.loads, open(cf))}; ids = json.load(open(f"{QS}/acts/{s}_meta.json"))["pair_ids"]
        k = [i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")]
        X = np.load(f"{QS}/acts/{s}_L48.npy", mmap_mode="r")[k, 8].astype(np.float64); R = [C[ids[i]] for i in k]
        D[s] = (X, np.array([int(r["jailbroken"]) for r in R]), R)
    names = json.load(open(f"{QS}/in/trait_names_layer32.json"))
    T = np.stack([torch.load(f"{QS}/in/trait_vectors/{n}.pt", map_location="cpu")[48].float().numpy() for n in names]).astype(np.float64)
    return D, T
def load_llama():
    import torch
    D = {}
    for s in SETS:
        d = {"hb": "harmbench", "wjb": "wildjailbreak"}.get(s, s)
        f = f"{LD}/{d}/classified_responses.jsonl"; f = f if os.path.exists(f) else f"{LD}/{d}/responses.jsonl"
        R = [json.loads(l) for l in open(f)]; A = torch.load(f"{LD}/{d}/activations.pt", map_location="cpu", weights_only=False)
        R = [r for r in R if r.get("jailbroken") is not None and r["pair_id"] in A and (s != "hb" or r["attack_type"] == "human_jailbreak")]
        D[s] = (np.stack([A[r["pair_id"]]["16"].float().numpy() for r in R]).astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R)
    return D, np.load(f"{LD}/trait_matrix_layer16.npy").astype(np.float64)

def fold(y, s):
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, s); return max(a, 1 - a)

def val_split(R, y, hb_only):
    for seed in range(300):
        rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in R}); rng.shuffle(beh); vb = set(beh[:int(0.2 * len(beh))])
        if hb_only:
            tp = sorted({r["jailbreak_idx"] for r in R}); rng.shuffle(tp); vt = set(tp[:int(0.2 * len(tp))])
            va = [i for i, r in enumerate(R) if r["behavior_id"] in vb and r["jailbreak_idx"] in vt]; tr = [i for i, r in enumerate(R) if r["behavior_id"] not in vb and r["jailbreak_idx"] not in vt]
        else:
            va = [i for i, r in enumerate(R) if r["behavior_id"] in vb]; tr = [i for i, r in enumerate(R) if r["behavior_id"] not in vb]
        va, tr = np.array(va), np.array(tr)
        if 0 < y[va].sum() < len(va) and 0 < y[tr].sum() < len(tr): return tr, va
    raise RuntimeError("no val split")

# ---- methods: each = list of configs; fit(cfg, X, y, w) -> scorer(Xtest) -> scores. Features chosen per method. ----
def make_methods(T, Gs):
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier; from sklearn.neural_network import MLPClassifier
    P = lambda V: (lambda X: X @ V.T); raw = lambda X: X
    def lr(feat):
        def fit(C, X, y, w):
            Z = feat(X); sc = StandardScaler().fit(Z); m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Z), y, sample_weight=w)
            return lambda Xt: m.decision_function(sc.transform(feat(Xt)))
        return fit
    def pca_lr(cfg, X, y, w):
        k, C = cfg; sc = StandardScaler().fit(X); p = PCA(k, random_state=0).fit(sc.transform(X)); m = LogisticRegression(C=C, max_iter=5000).fit(p.transform(sc.transform(X)), y, sample_weight=w)
        return lambda Xt: m.decision_function(p.transform(sc.transform(Xt)))
    def dmu(std):
        def fit(_, X, y, w):
            mu = X.mean(0); sd = X.std(0) + 1e-6 if std else 1.0; Z = (X - mu) / sd
            d = np.average(Z[y == 1], 0, w[y == 1]) - np.average(Z[y == 0], 0, w[y == 0])
            return lambda Xt: ((Xt - mu) / sd) @ d
        return fit
    concat = lambda X: np.hstack([X @ T.T, X])
    def hgb(feat_kind):
        def fit(cfg, X, y, w):
            lr_, leaves = cfg
            if feat_kind == "pj": f = P(T)
            else:
                sc = StandardScaler().fit(X); p = PCA(128, random_state=0).fit(sc.transform(X)); f = lambda Z: p.transform(sc.transform(Z))
            m = HistGradientBoostingClassifier(learning_rate=lr_, max_leaf_nodes=leaves, max_iter=300, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0).fit(f(X), y, sample_weight=w)
            return lambda Xt: m.predict_proba(f(Xt))[:, 1]
        return fit
    def mlp(feat):
        def fit(alpha, X, y, w):
            Z = feat(X); sc = StandardScaler().fit(Z)
            idx = np.random.default_rng(0).choice(len(y), size=len(y), p=w / w.sum()) if not np.allclose(w, w[0]) else np.arange(len(y))   # MLP has no sample_weight: weighted resample
            m = MLPClassifier(hidden_layer_sizes=(128,), alpha=alpha, early_stopping=True, validation_fraction=0.15, max_iter=300, random_state=0).fit(sc.transform(Z[idx]), y[idx])
            return lambda Xt: m.predict_proba(sc.transform(feat(Xt)))[:, 1]
        return fit
    M = {"LR_PersonaJB": (lr(P(T)), CG), **{f"LR_Gaussian{g}": (lr(P(G)), CG) for g, G in enumerate(Gs)}, "LR_raw": (lr(raw), CG),
         "PCA_LR": (pca_lr, [(k, C) for k in KS for C in CG]), "DiffMeans_raw": (dmu(False), [None]), "DiffMeans_std": (dmu(True), [None]),
         "Concat_LR": (lr(concat), CG), "HGB_PersonaJB": (hgb("pj"), [(a, b) for a in (0.05, 0.1) for b in (15, 31)]),
         "HGB_PCA128": (hgb("pca"), [(a, b) for a in (0.05, 0.1) for b in (15, 31)]), "MLP_PersonaJB": (mlp(P(T)), [1e-4, 1e-2, 1.0]), "MLP_raw": (mlp(raw), [1e-4, 1e-2, 1.0])}
    return M

def run_cell(model, regime, train_sets, tests, mname, D, T, Gs):
    M = make_methods(T, Gs); fit, cfgs = M[mname]
    X = np.vstack([D[s][0] for s in train_sets]); y = np.concatenate([D[s][1] for s in train_sets]); R = [r for s in train_sets for r in D[s][2]]
    w = np.concatenate([np.full(len(D[s][1]), len(y) / len(train_sets) / len(D[s][1])) for s in train_sets])   # each training set equal total weight
    tr, va = val_split(R, y, train_sets == ["hb"])
    vauc, vsc = {}, {}
    for c in cfgs:
        try: s = fit(c, X[tr], y[tr], w[tr])(X[va]); vauc[str(c)] = fold(y[va], s); vsc[str(c)] = s.astype(np.float32)
        except Exception as e: vauc[str(c)] = float("nan")
    best = max(cfgs, key=lambda c: -1 if np.isnan(vauc[str(c)]) else vauc[str(c)])
    sc = fit(best, X, y, w)
    return (model, regime, mname), {"cfg": str(best), "val_auc": vauc, "val_idx": va, "val_scores": vsc, "test": {t: sc(D[t][0]).astype(np.float32) for t in tests}}

if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    res, SC = {}, {}
    for model, loader in (("qwen", load_qwen), ("llama", load_llama)):
        D, T = loader(); T /= np.linalg.norm(T, axis=1, keepdims=True)
        Gs = [np.random.default_rng(g).standard_normal(T.shape) for g in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
        print(model, {s: (len(D[s][1]), int(D[s][1].sum())) for s in SETS}, flush=True)
        regimes = [(f"single_{a}", [a], [f for f in FAM if f != a]) for a in SETS] + [(f"lofo_{f}", [s for s in SETS if s != f], [f]) for f in FAM]
        mnames = list(make_methods(T, Gs).keys())
        out = dict(Parallel(n_jobs=NJ, backend="loky", verbose=5)(delayed(run_cell)(model, rg, trs, tes, m, D, T, Gs) for rg, trs, tes in regimes for m in mnames))
        behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {b: i for i, b in enumerate(behs)}
        W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B); bidx = {s: np.array([bi[r["behavior_id"]] for r in D[s][2]]) for s in SETS}
        def boot(t, s):
            y = D[t][1]; sg = 1 if roc_auc_score(y, s) >= 0.5 else -1; ww = W[:, bidx[t]]
            bs = np.array([roc_auc_score(y, sg * s, sample_weight=ww[k]) if (ww[k] * y).sum() > 0 and (ww[k] * (1 - y)).sum() > 0 else np.nan for k in range(B)])
            return roc_auc_score(y, sg * s), bs
        for (mo, rg, m), v in out.items():
            pts, bss = zip(*[boot(t, s) for t, s in v["test"].items()])
            res[f"{mo}__{rg}__{m}"] = {"cfg": v["cfg"], "val_auc": v["val_auc"], "auc": float(np.mean(pts)), "per_test": dict(zip(v["test"], map(float, pts))), "boot": np.nanmean(np.stack(bss), 0)}
            p = f"{mo}__{rg}__{m}"; SC[f"{p}__val_idx"] = v["val_idx"]
            for c, s in v["val_scores"].items(): SC[f"{p}__val__{c}"] = s
            for t, s in v["test"].items(): SC[f"{p}__test_{t}"] = s
        for s in SETS: SC[f"{model}__labels_{s}"] = D[s][1]; SC[f"{model}__behavior_ids_{s}"] = np.array([r["behavior_id"] for r in D[s][2]]); SC[f"{model}__pair_ids_{s}"] = np.array([str(r["pair_id"]) for r in D[s][2]])
        print(model, "done", flush=True)
    # aggregate: Gaussian = mean of draws; averages over single regimes and over lofo regimes
    models = ["llama", "qwen"]; rgs = [f"single_{a}" for a in SETS] + [f"lofo_{f}" for f in FAM]
    meths = ["LR_PersonaJB", "LR_Gaussian", "LR_raw", "PCA_LR", "DiffMeans_raw", "DiffMeans_std", "Concat_LR", "HGB_PersonaJB", "HGB_PCA128", "MLP_PersonaJB", "MLP_raw"]
    def get(mo, rg, m):
        if m == "LR_Gaussian": return np.mean([res[f"{mo}__{rg}__LR_Gaussian{g}"]["auc"] for g in range(3)]), np.mean([res[f"{mo}__{rg}__LR_Gaussian{g}"]["boot"] for g in range(3)], 0)
        return res[f"{mo}__{rg}__{m}"]["auc"], res[f"{mo}__{rg}__{m}"]["boot"]
    nb = len(models) * len(meths) * 2; q = 100 * 0.05 / nb / 2; table = {}
    for mo in models:
        for m in meths:
            for name, sel in (("single_avg", [r for r in rgs if r.startswith("single")]), ("lofo_avg", [r for r in rgs if r.startswith("lofo")])) + tuple((r, [r]) for r in rgs):
                pt = np.mean([get(mo, r, m)[0] for r in sel]); bs = np.mean([get(mo, r, m)[1] for r in sel], 0)
                pr = np.mean([get(mo, r, "LR_raw")[0] for r in sel]); br = np.mean([get(mo, r, "LR_raw")[1] for r in sel], 0)
                table[f"{mo}__{m}__{name}"] = {"auc": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                              "delta_vs_LR_raw": float(pt - pr), "delta_ci_bonf": [float(np.nanpercentile(bs - br, q)), float(np.nanpercentile(bs - br, 100 - q))],
                                              "cfg": None if len(sel) > 1 or m == "LR_Gaussian" else res[f"{mo}__{sel[0]}__{m}"]["cfg"]}
    for k, v in res.items(): SC[f"boot__{k}"] = v.pop("boot").astype(np.float32)
    json.dump({"B": B, "bonferroni_m": nb, "table": table, "cells": res}, open(f"{OUT}/probe_methods.json", "w"), indent=1)
    np.savez(f"{OUT}/scores_probe_methods.npz", **SC)
    for mo in models:
        print(f"\n== {mo}: method | single-set avg [95% CI] (delta vs LR_raw, Bonferroni CI) | LOFO avg [95% CI] (delta)")
        for m in meths:
            a, b = table[f"{mo}__{m}__single_avg"], table[f"{mo}__{m}__lofo_avg"]
            print(f"{m:14s} {a['auc']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}] ({a['delta_vs_LR_raw']:+.3f} [{a['delta_ci_bonf'][0]:+.3f},{a['delta_ci_bonf'][1]:+.3f}]) | "
                  f"{b['auc']:.3f} [{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}] ({b['delta_vs_LR_raw']:+.3f} [{b['delta_ci_bonf'][0]:+.3f},{b['delta_ci_bonf'][1]:+.3f}]) | LOFO per family " +
                  " ".join(f"{f} {table[f'{mo}__{m}__lofo_{f}']['auc']:.3f}" for f in FAM), flush=True)
