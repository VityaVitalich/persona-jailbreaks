#!/usr/bin/env python3
"""PersonaJB detection for Qwen3-32B on the node CPU (standalone; same recipe as trait_pc_sweep.py, Gemma protocol):
StandardScaler + LR (C=10, lbfgs, max_iter 4000) on the projections onto the 240 trait vectors (layer 32, unit rows),
fit on the HarmBench human_jailbreak train pool plus all WildJailbreak rows (if collected); 50 pool splits (pool_split
copied from trait_pc_sweep: 70% of behaviours x 70% of templates train, the disjoint rest test, random.Random(seed));
folded AUC on the HarmBench test pool (splits with both classes) and on every collected transfer family.
Reference dictionary: 240 random Gaussian unit directions (3 draws), same detector.
Saves every per-prompt score (split x set, PersonaJB and Gaussian draw 0) to scores.npz and the table to detection.json.
Transfer families: every collected one of GCG, PAIR, PAP, GPTFuzz, PEZ (GCG/PAIR/PEZ/GPTFuzz = test cases optimised against
Gemma-4-31B, transferred to Qwen). usage: qwen_detect.py OUTDIR trait_matrix_layer32.npy [OUTPUT_PREFIX]"""
import json, os, random, sys, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
od, tm = sys.argv[1:3]; TAG = sys.argv[3] if len(sys.argv) > 3 else ""
def load(st):
    rows = [json.loads(l) for l in open(f"{od}/classified_{st}.jsonl")]; A = torch.load(f"{od}/acts_{st}.pt", map_location="cpu")
    rows = [r for r in rows if r["jailbroken"] is not None and r["pair_id"] in A]
    return np.stack([A[r["pair_id"]]["32"].float().numpy() for r in rows]).astype(np.float64), np.array([int(r["jailbroken"]) for r in rows]), rows
def pool_split(rows, seed, f=0.7):
    rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in rows}); tpl = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(beh); rng.shuffle(tpl); nb, nt = max(1, int(len(beh) * f)), max(1, int(len(tpl) * f))
    trb, trt, teb, tet = set(beh[:nb]), set(tpl[:nt]), set(beh[nb:]), set(tpl[nt:])
    return (np.array([i for i, r in enumerate(rows) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]),
            np.array([i for i, r in enumerate(rows) if r["behavior_id"] in teb and r["jailbreak_idx"] in tet]))
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
mk = lambda: Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=10.0, penalty="l2", solver="lbfgs", max_iter=4000, random_state=0))])
Xh, yh, hv = load("hb"); k = [i for i, r in enumerate(hv) if r["attack_type"] == "human_jailbreak"]; Xh, yh, hv = Xh[k], yh[k], [hv[i] for i in k]
Xa, ya = (load("wjb")[:2] if os.path.exists(f"{od}/classified_wjb.jsonl") and os.path.exists(f"{od}/acts_wjb.pt") else (np.zeros((0, Xh.shape[1])), np.zeros(0, int)))
fam = {s.upper(): load(s)[:2] for s in ("gcg", "pair", "pap", "gptfuzz", "pez") if os.path.exists(f"{od}/classified_{s}.jsonl") and os.path.exists(f"{od}/acts_{s}.pt")}
T = np.load(tm).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
splits = [pool_split(hv, s) for s in range(50)]; splits = [(a, b) for a, b in splits if len(a) and 0 < yh[b].sum() < len(b) and 0 < yh[a].sum() < len(a)]
if not splits:
    json.dump({"error": "no pool split with both classes in train and test", "n_hb": len(yh), "hb_jb": int(yh.sum())}, open(f"{od}/detection.json", "w")); print("no valid split; wrote detection.json with error"); sys.exit(0)
def detector(V, save=None):
    hb, xf, sc = [], {f: [] for f in fam}, {}
    for j, (tr, te) in enumerate(splits):
        m = mk().fit(np.vstack([Xh[tr] @ V.T, Xa @ V.T]), np.r_[yh[tr], ya]); s = m.decision_function(Xh @ V.T); hb.append(fold(yh[te], s[te]))
        if save is not None: sc[f"{save}__HB__s{j}"] = s.astype(np.float32); sc[f"idx_train__s{j}"] = tr; sc[f"idx_test__s{j}"] = te
        for f, (X, y) in fam.items():
            sf = m.decision_function(X @ V.T); xf[f].append(fold(y, sf))
            if save is not None: sc[f"{save}__{f}__s{j}"] = sf.astype(np.float32)
    return dict(hb=float(np.mean(hb)), hb_ci95=float(1.96 * np.std(hb, ddof=1) / np.sqrt(len(hb))), **{f: float(np.mean(v)) for f, v in xf.items()},
                transfer_mean=float(np.mean([np.mean(v) for v in xf.values()])) if xf else None), sc
res = {"n": {"hb_human_jailbreak": len(yh), "hb_jb": int(yh.sum()), "wjb": len(ya), "wjb_jb": int(ya.sum()), **{f: [len(y), int(y.sum())] for f, (X, y) in fam.items()}},
       "splits_used": len(splits), "transfer_families": list(fam),
       "note": "GCG, PAIR, PEZ, GPTFuzz: test cases optimised against Gemma-4-31B, transferred to Qwen; PAP: target-agnostic (olmo3_7b_top5)"}
r, sc = detector(T, save="personajb"); res["PersonaJB"] = r; print("PersonaJB", r, flush=True)
g = []
for i in range(3):
    G = np.random.default_rng(i).standard_normal(T.shape); G /= np.linalg.norm(G, axis=1, keepdims=True)
    rr, s2 = detector(G, save="gauss0" if i == 0 else None); g.append(rr); sc.update(s2)
res["Random Gaussian (3 draws)"] = {k2: (float(np.mean([x[k2] for x in g])) if g[0][k2] is not None else None) for k2 in g[0]}
print("Random Gaussian", res["Random Gaussian (3 draws)"], flush=True)
np.savez(f"{od}/{TAG}scores.npz", labels_HB=yh, pair_ids_HB=np.array([r["pair_id"] for r in hv]), **{f"labels_{f}": y for f, (X, y) in fam.items()}, **sc)
json.dump(res, open(f"{od}/{TAG}detection.json", "w"), indent=1)
