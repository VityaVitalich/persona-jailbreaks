#!/usr/bin/env python3
"""Is there a jailbreak *region* in a low-dimensional persona space (top-k trait PCs)?
 (A) nonlinear detectors on top-k centered trait PCs, PersonaJB protocol (HB train pool [+WJB on Gemma] -> HB test + 5 families),
     20 seeds. logreg (C=10, reference) / QDA (reg 0.1) / RBF-SVM / HistGradientBoosting / kNN(25), all on standardized features.
 (B) within-family: 5-fold StratifiedGroupKFold by behavior_id x 4 repeats inside each attack family; does a top-k PC
     detector fit *that family*? logreg C=1, plus HGB at k=10. Then cosine between the families' k=10 logreg normals.
Per-prompt scores saved to region_tests_<model>.npz."""
import json, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
sys.path.insert(0, "."); import trait_pc_sweep as S
FAMS = S.FAMS
CLFS = {"logreg": lambda s: LogisticRegression(C=10, max_iter=4000, random_state=s),
        "qda": lambda s: QDA(reg_param=0.1),
        "rbf_svm": lambda s: SVC(C=1, gamma="scale", random_state=s),
        "hgb": lambda s: HGB(random_state=s),
        "knn25": lambda s: KNeighborsClassifier(25)}
def score(m, X): return m.decision_function(X) if hasattr(m, "decision_function") and not hasattr(m, "predict_proba") or isinstance(m[-1], SVC) else m.predict_proba(X)[:, 1]
def sym(y, s):
    if len(np.unique(y)) < 2: return np.nan, np.nan
    a = roc_auc_score(y, s); return a, max(a, 1-a)
m = sys.argv[1]; cfg = S.CFG[m]; L = cfg["layer"]; t0 = time.time()
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
_, Sc, Vc = np.linalg.svd(T - T.mean(0), full_matrices=False)
rows = {}; data = {}
hr = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
X, y, v = S.build_matrix(hr, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), L); data["HB"] = (X, y, v)
for f, (rp, ap) in cfg["fams"].items():
    data[f] = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), L)
aug = S.build_matrix(S.load_jsonl(cfg["aug"][0]), torch.load(cfg["aug"][1], map_location="cpu", weights_only=False), L) if cfg["aug"] else None
feats = {f"pc{k}": Vc[:k].T for k in (5, 10, 20)}; feats["traits"] = T.T
F = {fn: {s: (d[0] @ W).astype(np.float32) for s, d in data.items()} for fn, W in feats.items()}
Faug = {fn: (aug[0] @ W).astype(np.float32) for fn, W in feats.items()} if aug else None
out = {"model": m, "A": {}, "B": {}}; npz = {}
for s, d in data.items(): npz[f"labels__{s}"] = d[1]; npz[f"pair_ids__{s}"] = np.array([r["pair_id"] for r in d[2]])
# ---- (A)
hv = data["HB"][2]; yh = data["HB"][1]; NS = 20
splits = [(s, *S.pool_split(hv, s)) for s in range(NS)]
for fn in F:
    for cn, mk in CLFS.items():
        if fn == "traits" and cn in ("qda", "knn25"): continue
        res = {k: [] for k in ["human_test"] + FAMS}; sc = {k: [] for k in ["HB"] + FAMS}
        for s, tr, te in splits:
            Xtr, ytr = F[fn]["HB"][tr], yh[tr]
            if Faug: Xtr, ytr = np.vstack([Xtr, Faug[fn]]), np.concatenate([ytr, aug[1]])
            mdl = make_pipeline(StandardScaler(), mk(s)).fit(Xtr, ytr)
            g = lambda Z: mdl.decision_function(Z) if cn == "rbf_svm" else mdl.predict_proba(Z)[:, 1]
            sh = g(F[fn]["HB"]); sc["HB"].append(sh); res["human_test"].append(sym(yh[te], sh[te])[1])
            for f in FAMS:
                st = g(F[fn][f]); sc[f].append(st); res[f].append(sym(data[f][1], st)[1])
        xf = np.nanmean([res[f] for f in FAMS], 0)
        out["A"][f"{fn}/{cn}"] = {"hb": float(np.nanmean(res["human_test"])), "xfer": float(xf.mean()), "ci": float(1.96*xf.std(ddof=1)/np.sqrt(len(xf))),
                                  "fam": {f: float(np.nanmean(res[f])) for f in FAMS}, "per_seed": res}
        for k, a in sc.items(): npz[f"A__{fn}__{cn}__{k}"] = np.array(a, np.float32)
        o = out["A"][f"{fn}/{cn}"]; print(f"A {fn:7s} {cn:8s} HB={o['hb']:.3f} xfer={o['xfer']:.3f}±{o['ci']:.3f} " + " ".join(f"{f}={o['fam'][f]:.2f}" for f in FAMS), flush=True)
# ---- (B)
featsB = {f"pc{k}": Vc[:k].T for k in (2, 5, 10, 20, 50)}; featsB["traits"] = T.T
for s, (X, y, v) in data.items():
    groups = np.array([r["behavior_id"] for r in v]); out["B"][s] = {}
    for fn, W in featsB.items():
        Z = (X @ W).astype(np.float32)
        for cn in (["logreg", "hgb"] if fn == "pc10" else ["logreg"]):
            aucs = []; oof_all = []
            for rep in range(4):
                oof = np.full(len(y), np.nan)
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(Z, y, groups):
                    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=4000) if cn == "logreg" else HGB(random_state=rep)).fit(Z[tr], y[tr])
                    oof[te] = mdl.predict_proba(Z[te])[:, 1]
                aucs.append(roc_auc_score(y, oof)); oof_all.append(oof)
            out["B"][s][f"{fn}/{cn}"] = float(np.mean(aucs)); npz[f"B__{s}__{fn}__{cn}"] = np.array(oof_all, np.float32)
    print(f"B {s:8s} n={len(y)} jb={int(y.sum())} | " + " ".join(f"{k}={a:.3f}" for k, a in out["B"][s].items()), flush=True)
W10 = {}
for s, (X, y, v) in data.items():
    Z = X @ Vc[:10].T; mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=4000)).fit(Z, y)
    w = mdl[-1].coef_[0] / mdl[0].scale_; W10[s] = w / np.linalg.norm(w)
names = list(W10); Cm = np.array([[W10[a] @ W10[b] for b in names] for a in names])
out["B_cos_pc10"] = {"names": names, "cos": Cm.tolist()}
print("cos between per-family normals in top-10 PC space:\n" + "        " + " ".join(f"{n:>8s}" for n in names))
for i, n in enumerate(names): print(f"{n:8s}" + " ".join(f"{c:8.2f}" for c in Cm[i]))
json.dump(out, open(f"region_tests_{m}.json", "w")); np.savez_compressed(f"region_tests_{m}.npz", **npz)
print(f"done {time.time()-t0:.0f}s")
