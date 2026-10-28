#!/usr/bin/env python3
"""
Trait-space PC sweep for the PersonaJB detector (persona-jailbreaks paper).

PersonaJB = logistic regression on the trait projections z = T h (T: n_traits x d, unit rows).
Here the n trait directions are replaced by the top-k principal components of the trait
matrix itself, z_k = U_k^T h, and k is swept from 1 up to the rank of T. Nothing about the
data enters the basis; only the trait vectors do.

Two bases:
  pca  centered PCA over traits: right singular vectors of (T - mean_trait). Max rank n-1;
       the mean trait direction itself is not in the span.
  pcamean  the mean trait direction + the top k-1 centered PCs (k dims in total); at full
       k the span equals span(T). Separates what centering removes.
  svd  uncentered: right singular vectors of T (the "trait PCs" of the old dose-response
       appendix table). At k = n the span equals span(T) exactly.
plus  traits  the PersonaJB features themselves (k = n), a reproduction check against
       the published row.

Protocol = the published PersonaJB row (run_all_traits_sweep_v2.py, spec logreg_l2_C10.0_raw):
  per seed in 0..49: behaviour x template pool split (train_frac 0.7) of HarmBench human
  jailbreaks; inner stratified val split (15%, random_state=seed) of the train pool;
  Pipeline(StandardScaler, LogisticRegression(C=10, l2, lbfgs, max_iter=4000, rs=seed)).
  Val scores come from the model fit on train-minus-val; test and transfer scores from the
  model refit on the whole train pool. Gemma folds all WildJailbreak rows into both fits
  (eval sets stay pure HarmBench / attack families). Cell AUC = max(auc, 1-auc) (safe_auc);
  signed AUC is stored too. Avg transfer = mean over GCG/PAIR/PAP/GPTFuzz/PEZ; CI =
  1.96 * std / sqrt(n_seeds) of the per-seed averages.

Saves everything per prompt (npz): for every basis x k x seed, the score of every HarmBench
row from the val-fit model and from the refit model, and of every transfer row from the refit
model; plus pair_ids, labels and the per-seed train/val/test indices.
"""
import argparse, json, random, sys, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious macOS Accelerate matmul FP warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path.home() / "persona-vector-jbing"
OUT = REPO / "full_trait_output"
HERE = OUT / "trait_pc_sweep"
FAMS = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
C = 10.0  # PersonaJB row; --C overrides (regularization diagnostic only)

CFG = {
    "llama": dict(
        layer=16, T=OUT/"trait_matrix_layer16.npy",
        human=(OUT/"harmbench_activations/classified_responses.jsonl", OUT/"harmbench_activations/activations.pt"),
        fams={f: (OUT/f"{f.lower()}_activations/responses.jsonl", OUT/f"{f.lower()}_activations/activations.pt") for f in FAMS},
        aug=None, published={"human_test": .8538, "GCG": .8166, "PAIR": .7900, "PAP": .6413, "GPTFuzz": .7083, "PEZ": .8416, "xfer": .7596}),
    "gemma": dict(
        layer=30, T=HERE/"_gemma_ref/trait_matrix_layer30.npy",  # cluster copy, 240 rows (local mirror's is stale)
        human=(OUT/"harmbench_activations_gemma/classified_responses.jsonl", OUT/"harmbench_activations_gemma/activations.pt"),
        fams={f: (OUT/f"{f.lower()}_activations_gemma_hb/classified_responses.jsonl", OUT/f"{f.lower()}_activations_gemma_hb/activations.pt") for f in FAMS},
        aug=(OUT/"wildjailbreak_activations_gemma/classified_responses.jsonl", OUT/"wildjailbreak_activations_gemma/activations.pt"),
        published={"human_test": .8773, "GCG": .7383, "PAIR": .6782, "PAP": .6937, "GPTFuzz": .7590, "PEZ": .8402, "xfer": .7419}),
}

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def build_matrix(rows, acts, layer):  # = run_all_traits_sweep_v2.build_activation_matrix
    lk = str(layer); X, y, valid = [], [], []
    for r in rows:
        pid, jb = r.get("pair_id"), r.get("jailbroken")
        if pid is None or jb is None: continue
        it = acts.get(pid)
        if it is None or lk not in it: continue
        X.append(it[lk].float().numpy()); y.append(1 if jb else 0); valid.append(r)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), valid

def pool_split(rows, seed):  # = get_pool_split + split_by_pool
    rng = random.Random(seed)
    beh = sorted({r["behavior_id"] for r in rows}); tpl = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(beh); rng.shuffle(tpl)
    nb, nt = max(1, int(len(beh)*TRAIN_FRAC)), max(1, int(len(tpl)*TRAIN_FRAC))
    trb, trt, teb, tet = set(beh[:nb]), set(tpl[:nt]), set(beh[nb:]), set(tpl[nt:])
    tr = [i for i, r in enumerate(rows) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]
    te = [i for i, r in enumerate(rows) if r["behavior_id"] in teb and r["jailbreak_idx"] in tet]
    return tr, te

def auc2(y, s):
    if len(np.unique(y)) < 2: return float("nan"), float("nan")
    a = float(roc_auc_score(y, s)); return a, max(a, 1-a)

def model(seed):
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=4000, class_weight=None, random_state=seed))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CFG, required=True)
    ap.add_argument("--ks", default="1,2,3,4,5,6,8,10,12,16,20,24,32,48,64,96,128,160,192,full")
    ap.add_argument("--bases", default="pca,pcamean,svd,traits")
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--extra_bases", default=None, help="npz of named [r x d] bases (rows ordered), e.g. lda_bases_llama.npz")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    global C; C = a.C
    cfg = CFG[a.model]; L = cfg["layer"]; t0 = time.time()

    T = np.load(cfg["T"]).astype(np.float64)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    n = T.shape[0]
    Tc = T - T.mean(0, keepdims=True)
    _, Sc, Vc = np.linalg.svd(Tc, full_matrices=False)
    _, Su, Vu = np.linalg.svd(T, full_matrices=False)
    rank_c, rank_u = int((Sc > 1e-8*Sc[0]).sum()), int((Su > 1e-8*Su[0]).sum())
    evr = {"pca": np.cumsum(Sc**2)/np.sum(Sc**2), "svd": np.cumsum(Su**2)/np.sum(Su**2)}
    print(f"[{a.model}] T {T.shape}; rank centered {rank_c}, uncentered {rank_u}", flush=True)
    mu = T.mean(0); mu_c = mu - Vc[:rank_c].T @ (Vc[:rank_c] @ mu); mu_c /= np.linalg.norm(mu_c)  # mean dir is orthogonal to centered PCs up to fp
    basis = {"pca": (Vc[:rank_c], rank_c), "svd": (Vu[:rank_u], rank_u),
             "pcamean": (np.vstack([mu_c[None], Vc[:rank_c]]), rank_c + 1)}  # [mean trait dir, PC1, PC2, ...]
    evr["pcamean"] = np.concatenate([[np.nan], evr["pca"]])  # entry k-1 = cum. centered var of the k-1 PCs

    hrows = [r for r in load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    Xh, yh, hvalid = build_matrix(hrows, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), L)
    sets = {"HB": (Xh, yh, [r["pair_id"] for r in hvalid])}
    for f, (rp, apth) in cfg["fams"].items():
        X, y, v = build_matrix(load_jsonl(rp), torch.load(apth, map_location="cpu", weights_only=False), L)
        sets[f] = (X, y, [r["pair_id"] for r in v])
    Xa = ya = None
    if cfg["aug"]:
        Xa, ya, va = build_matrix(load_jsonl(cfg["aug"][0]), torch.load(cfg["aug"][1], map_location="cpu", weights_only=False), L)
        sets["AUG_train_only"] = (Xa, ya, [r["pair_id"] for r in va])
    for s, (X, y, _) in sets.items(): print(f"  {s}: {X.shape} jb={int(y.sum())}", flush=True)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    splits = []
    for seed in range(a.n_seeds):
        tr, te = pool_split(hvalid, seed)
        if not tr or not te or len(np.unique(yh[tr])) < 2: continue
        tr_i, va_i = train_test_split(np.array(tr), test_size=VAL_FRAC, random_state=seed, stratify=yh[tr])
        splits.append((seed, np.array(tr), tr_i, va_i, np.array(te)))
    print(f"  {len(splits)} seeds; train ~{len(splits[0][1])} (fit {len(splits[0][2])} / val {len(splits[0][3])}), test ~{len(splits[0][4])}", flush=True)

    if a.extra_bases:
        for nm, V in np.load(a.extra_bases).items():
            basis[nm] = (V.astype(np.float64), V.shape[0]); evr[nm] = np.full(V.shape[0], np.nan)
    configs = []  # (basis, k_requested, k_used, W [d x k])
    for b in a.bases.split(","):
        if b == "traits": configs.append(("traits", n, n, T.T)); continue
        V, r = basis[b]
        for ks in a.ks.split(","):
            k = r if ks == "full" else min(int(ks), r)
            if any(c[0] == b and c[2] == k for c in configs): continue
            configs.append((b, k, k, V[:k].T))

    nS = len(splits)
    scores = {f"{b}_k{k}": {"HB_valfit": np.full((nS, len(yh)), np.nan, np.float32),
                            "HB_refit": np.full((nS, len(yh)), np.nan, np.float32),
                            **{f: np.full((nS, len(sets[f][1])), np.nan, np.float32) for f in FAMS}}
              for b, _, k, _ in configs}
    summary = {"model": a.model, "layer": L, "n_traits": n, "rank_centered": rank_c, "rank_uncentered": rank_u,
               "C": C, "protocol": "run_all_traits_sweep_v2 logreg_l2_C10.0_raw (PersonaJB row); safe_auc; 50 seeds",
               "published_personajb": cfg["published"], "configs": {}}
    for b, _, k, W in configs:
        tk = time.time(); W = W.astype(np.float32)
        Z = {s: (X @ W) for s, (X, _, _) in sets.items()}
        res = {s: {"auc": [], "auc_signed": []} for s in ["val", "human_test"] + FAMS}
        S = scores[f"{b}_k{k}"]
        for si, (seed, tr, tr_i, va_i, te) in enumerate(splits):
            def fitset(idx):
                if Xa is None: return Z["HB"][idx], yh[idx]
                return np.vstack([Z["HB"][idx], Z["AUG_train_only"]]), np.concatenate([yh[idx], ya])
            m1 = model(seed).fit(*fitset(tr_i)); S["HB_valfit"][si] = m1.predict_proba(Z["HB"])[:, 1]
            m2 = model(seed).fit(*fitset(tr));   S["HB_refit"][si] = m2.predict_proba(Z["HB"])[:, 1]
            for f in FAMS: S[f][si] = m2.predict_proba(Z[f])[:, 1]
            for key, y, s in [("val", yh[va_i], S["HB_valfit"][si][va_i]), ("human_test", yh[te], S["HB_refit"][si][te])] + \
                             [(f, sets[f][1], S[f][si]) for f in FAMS]:
                sg, sy = auc2(y, s); res[key]["auc"].append(sy); res[key]["auc_signed"].append(sg)
        xf = np.nanmean(np.array([res[f]["auc"] for f in FAMS]), 0)
        xs = np.nanmean(np.array([res[f]["auc_signed"] for f in FAMS]), 0)
        ev = float(evr[b][k-1]) if b in evr else 1.0
        summary["configs"][f"{b}_k{k}"] = {"basis": b, "k": k, "trait_var_explained": ev, "per_seed": res,
            "mean": {s: float(np.nanmean(v["auc"])) for s, v in res.items()},
            "mean_signed": {s: float(np.nanmean(v["auc_signed"])) for s, v in res.items()},
            "avg_transfer": float(xf.mean()), "avg_transfer_ci95": float(1.96*xf.std(ddof=1)/np.sqrt(len(xf))),
            "avg_transfer_signed": float(xs.mean())}
        c = summary["configs"][f"{b}_k{k}"]
        print(f"  {b:6s} k={k:4d} ev={ev:.3f} | val={c['mean']['val']:.3f} HB={c['mean']['human_test']:.3f} " +
              " ".join(f"{f}={c['mean'][f]:.3f}" for f in FAMS) +
              f" | xfer={c['avg_transfer']:.3f}±{c['avg_transfer_ci95']:.3f} signed={c['avg_transfer_signed']:.3f} | {time.time()-tk:.0f}s", flush=True)
        json.dump(summary, open(HERE/f"summary_{a.model}{a.tag}.json", "w"))

    npz = {"seeds": np.array([s[0] for s in splits])}
    for s, (_, y, pids) in sets.items():
        npz[f"labels__{s}"] = y; npz[f"pair_ids__{s}"] = np.array(pids, dtype=object)
    for si, (seed, tr, tr_i, va_i, te) in enumerate(splits):
        npz[f"idx_train__s{seed}"], npz[f"idx_fit__s{seed}"], npz[f"idx_val__s{seed}"], npz[f"idx_test__s{seed}"] = tr, tr_i, va_i, te
    for cname, d in scores.items():
        for s, arr in d.items(): npz[f"scores__{cname}__{s}"] = arr
    np.savez_compressed(HERE/f"scores_{a.model}{a.tag}.npz", **npz)
    print(f"done in {time.time()-t0:.0f}s -> {HERE}/summary_{a.model}{a.tag}.json, scores_{a.model}{a.tag}.npz", flush=True)

if __name__ == "__main__":
    main()
