#!/usr/bin/env python3
"""
PCA-dimensionality sweep for the "PCA (k)" detection baseline of the persona-jailbreaks paper.

Two protocols, each matching how the paper row was actually produced:

  llama  (fast_transfer_classifier.py for HB-test/GCG/PAIR; run_transfer_classifier.py for PAP/GPTFuzz/PEZ;
          the paper row ran with n_pca=4 although labelled "PCA (3)"):
     per seed: pool split (behaviour x template, train_frac 0.7) -> StandardScaler(train)
               -> PCA(k, random_state=42) fit on standardized train -> LogisticRegression(C=1, max_iter=2000, rs=42).
     Paper protocol per cell:  HB-test/GCG/PAIR = plain ROC-AUC on the full set;
                               PAP/GPTFuzz/PEZ  = max(AUC, 1-AUC) on a 50/50 undersample (RandomState 42) of the set.

  gemma  (run_all_traits_sweep_v2.py --feature_type pca --pca_n_components 3, row = logreg_l2_C1.0_bal):
     PCA(k, random_state=42) fit ONCE on all HarmBench human-jailbreak activations (unstandardized),
     per seed: pool split -> train pool = HB train + all WildJailbreak rows
               -> Pipeline(StandardScaler, LogisticRegression(C=1, l2, lbfgs, max_iter=4000,
                  class_weight='balanced', random_state=seed)).
     Paper protocol: AUC = max(auc, 1-auc) on the full set (sweep_v2's safe_auc).

Three AUC variants are stored per family and seed:
   auc       signed ROC-AUC on the full set (the honest number)
   auc_sym   max(auc, 1-auc) on the full set
   auc_paper the paper's cell protocol described above
k = 0 means "raw" (no PCA), a reference against the paper's Raw row.
"""
import argparse, json, os, random, sys, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious macOS Accelerate matmul FP warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn 1.8 'penalty' deprecation
from pathlib import Path
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path.home() / "persona-vector-jbing"
OUT  = REPO / "full_trait_output"
RS   = 42
TRAIN_FRAC = 0.7
FAMS = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
UNDERSAMPLED_IN_PAPER = {"llama": {"PAP", "GPTFuzz", "PEZ"}, "gemma": set()}

CFG = {
    "llama": dict(
        layer=16,
        human=(OUT/"harmbench_activations/classified_responses.jsonl", OUT/"harmbench_activations/activations.pt"),
        fams={f: (OUT/f"{f.lower()}_activations/responses.jsonl", OUT/f"{f.lower()}_activations/activations.pt") for f in FAMS},
        aug=None,
    ),
    "gemma": dict(
        layer=30,
        human=(OUT/"harmbench_activations_gemma/classified_responses.jsonl", OUT/"harmbench_activations_gemma/activations.pt"),
        fams={f: (OUT/f"{f.lower()}_activations_gemma_hb/classified_responses.jsonl", OUT/f"{f.lower()}_activations_gemma_hb/activations.pt") for f in FAMS},
        aug=(OUT/"wildjailbreak_activations_gemma/classified_responses.jsonl", OUT/"wildjailbreak_activations_gemma/activations.pt"),
    ),
}

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def build_matrix(rows, acts, layer):
    lk = str(layer); X, y, valid = [], [], []
    for r in rows:
        pid = r.get("pair_id"); jb = r.get("jailbroken")
        if pid is None or jb is None: continue
        it = acts.get(pid)
        if it is None or lk not in it: continue
        X.append(it[lk].float().numpy()); y.append(1 if jb else 0); valid.append(r)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), valid

def pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    beh = sorted({r["behavior_id"] for r in rows}); tpl = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(beh); rng.shuffle(tpl)
    nb = max(1, int(len(beh)*train_frac)); nt = max(1, int(len(tpl)*train_frac))
    trb, trt, teb, tet = set(beh[:nb]), set(tpl[:nt]), set(beh[nb:]), set(tpl[nt:])
    tr = [i for i, r in enumerate(rows) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]
    te = [i for i, r in enumerate(rows) if r["behavior_id"] in teb and r["jailbreak_idx"] in tet]
    return tr, te

def undersample_idx(y, seed=RS):
    """run_transfer_classifier.undersample_to_balanced: 50/50 subset of the test set, fixed seed."""
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]; n = min(len(ip), len(ineg))
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.choice(ip, n, replace=False), rng.choice(ineg, n, replace=False)])

def aucs(y, s, sub=None, paper_mode="signed"):
    """returns (signed, sym, paper)"""
    if len(np.unique(y)) < 2: return float("nan"), float("nan"), float("nan")
    a = float(roc_auc_score(y, s)); sym = max(a, 1-a)
    if paper_mode == "signed":      paper = a
    elif paper_mode == "sym":       paper = sym
    else:                           # "undersample_sym"
        b = float(roc_auc_score(y[sub], s[sub])); paper = max(b, 1-b)
    return a, sym, paper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CFG, required=True)
    ap.add_argument("--ks", default="0,1,2,3,4,5,6,8,10,12,16,20,24,32,48,64,96,128,192,256,384,512,768,1024")
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength (paper rows: 1.0)")
    ap.add_argument("--class_weight", choices=["paper", "balanced", "none"], default="paper",
                    help="paper = the row's own setting (llama none, gemma balanced)")
    a = ap.parse_args()
    cfg = CFG[a.model]; L = cfg["layer"]
    ks = [int(k) for k in a.ks.split(",")]
    us_fams = UNDERSAMPLED_IN_PAPER[a.model]
    hb_paper_mode = "signed" if a.model == "llama" else "sym"
    cw = {"paper": ("balanced" if a.model == "gemma" else None), "balanced": "balanced", "none": None}[a.class_weight]
    print(f"  classifier: L2 logistic regression, C={a.C}, class_weight={cw}", flush=True)

    t0 = time.time(); print(f"[{a.model}] loading", flush=True)
    hrows = [r for r in load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    Xh, yh, hvalid = build_matrix(hrows, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), L)
    print(f"  HB human: {Xh.shape}, jb={yh.sum()}", flush=True)
    fam, sub, pmode = {}, {}, {}
    for f, (rp, apth) in cfg["fams"].items():
        X, y, _ = build_matrix(load_jsonl(rp), torch.load(apth, map_location="cpu", weights_only=False), L)
        fam[f] = (X, y)
        if f in us_fams: sub[f] = undersample_idx(y); pmode[f] = "undersample_sym"
        else:            sub[f] = None;               pmode[f] = hb_paper_mode
        print(f"  {f}: {X.shape}, jb={y.sum()}  paper-cell protocol: {pmode[f]}" + (f" (n={len(sub[f])})" if sub[f] is not None else ""), flush=True)
    Xa = ya = None
    if cfg["aug"]:
        Xa, ya, _ = build_matrix(load_jsonl(cfg["aug"][0]), torch.load(cfg["aug"][1], map_location="cpu", weights_only=False), L)
        print(f"  AUG(train-only): {Xa.shape}, jb={ya.sum()}", flush=True)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    splits = [pool_split(hvalid, TRAIN_FRAC, s) for s in range(a.n_seeds)]
    splits = [(s, tr, te) for s, (tr, te) in enumerate(splits) if tr and te and len(np.unique(yh[tr])) > 1]
    print(f"  {len(splits)} usable seeds; train pool ~{len(splits[0][1])}, test ~{len(splits[0][2])}", flush=True)

    res = {"model": a.model, "layer": L, "n_seeds": a.n_seeds,
           "protocol": "fast_transfer + run_transfer_classifier(PAP/GPTFuzz/PEZ undersampled, sym)" if a.model == "llama" else "sweep_v2 logreg_l2_C1.0_bal (sym AUC)",
           "families": ["human_test"] + FAMS, "ks": ks, "C": a.C, "class_weight": cw, "per_k": {}}
    KEYS = ("auc", "auc_sym", "auc_paper")
    for k in ks:
        tk = time.time()
        per = {f: {kk: [] for kk in KEYS} for f in ["human_test"] + FAMS}
        evr = []
        def rec(f, trip):
            for kk, v in zip(KEYS, trip): per[f][kk].append(v)
        if a.model == "gemma":
            if k > 0:
                pca = PCA(n_components=min(k, Xh.shape[0]-1, Xh.shape[1]), random_state=RS)
                Zh = pca.fit_transform(Xh).astype(np.float32); evr.append(float(pca.explained_variance_ratio_.sum()))
                Zf = {f: pca.transform(X).astype(np.float32) for f, (X, _) in fam.items()}
                Za = pca.transform(Xa).astype(np.float32)
            else:
                Zh, Zf, Za = Xh, {f: X for f, (X, _) in fam.items()}, Xa
            for seed, tr, te in splits:
                Xtr = np.vstack([Zh[tr], Za]); ytr = np.concatenate([yh[tr], ya])
                m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=a.C, penalty="l2", solver="lbfgs", max_iter=4000, class_weight=cw, random_state=seed))])
                m.fit(Xtr, ytr)
                rec("human_test", aucs(yh[te], m.predict_proba(Zh[te])[:, 1], None, hb_paper_mode))
                for f, (X, y) in fam.items():
                    rec(f, aucs(y, m.predict_proba(Zf[f])[:, 1], sub[f], pmode[f]))
        else:
            for seed, tr, te in splits:
                sc = StandardScaler(); Xtr = sc.fit_transform(Xh[tr]); Xte = sc.transform(Xh[te])
                Xf = {f: sc.transform(X) for f, (X, _) in fam.items()}
                if k > 0:
                    pca = PCA(n_components=min(k, Xtr.shape[0]-1, Xtr.shape[1]), random_state=RS)
                    Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte); Xf = {f: pca.transform(X) for f, X in Xf.items()}
                    evr.append(float(pca.explained_variance_ratio_.sum()))
                clf = LogisticRegression(max_iter=2000, random_state=RS, C=a.C, class_weight=cw).fit(Xtr, yh[tr])
                rec("human_test", aucs(yh[te], clf.predict_proba(Xte)[:, 1], None, hb_paper_mode))
                for f, (X, y) in fam.items():
                    rec(f, aucs(y, clf.predict_proba(Xf[f])[:, 1], sub[f], pmode[f]))
        res["per_k"][str(k)] = {"explained_var": float(np.mean(evr)) if evr else 1.0, "per_family": per}
        mp = {f: np.nanmean(per[f]["auc_paper"]) for f in per}; msg = {f: np.nanmean(per[f]["auc"]) for f in per}
        xp = np.nanmean(np.array([per[f]["auc_paper"] for f in FAMS]), axis=0); xs = np.nanmean(np.array([per[f]["auc"] for f in FAMS]), axis=0)
        print(f"  k={k:5d} ev={res['per_k'][str(k)]['explained_var']:.3f} | PAPER: " + " ".join(f"{f}={mp[f]:.3f}" for f in per) +
              f" xfer={xp.mean():.3f}±{1.96*xp.std(ddof=1)/np.sqrt(len(xp)):.3f} | SIGNED: " + " ".join(f"{f}={msg[f]:.3f}" for f in per) +
              f" xfer={xs.mean():.3f} | {time.time()-tk:.0f}s", flush=True)
        json.dump(res, open(a.out, "w"))  # checkpoint after every k
    print(f"done in {time.time()-t0:.0f}s -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
