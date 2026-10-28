#!/usr/bin/env python3
"""Layer x position sweep of PersonaJB vs Gaussian vs full probe on Qwen3-32B (protocol of transfer/diag/diag_training.py):
StandardScaler + LR C=10 on 240 projections (PersonaJB: unit trait vectors at that layer; Gaussian: 3 unit random bases),
full probe LR C=0.01 on 5120 dims. Two training pools: HB human_jailbreak only, HB + all WJB. Metrics (folded AUC):
HB template/behaviour-disjoint test (20 splits; 5 for the full probe), HB random 5-fold CV (out-of-fold), and each
transfer family (GCG/PAIR/PAP/GPTFuzz/PEZ, Qwen's own test cases). One worker per layer; per-prompt scores for
PersonaJB, Gaussian0 and the full probe go to scores/L{l}_p{p}.npz, the table to sweep.json.
usage: detect_sweep.py SWEEPDIR TRANSFERDIR NJOBS"""
import json, os, random, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
D, TR, NJ = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.environ.setdefault("OMP_NUM_THREADS", "8")
from joblib import Parallel, delayed
FAM = ["gcg", "pair", "pap", "gptfuzz", "pez"]; SETS = ["hb", "wjb"] + FAM
CLS = {s: f"{D}/in/classified_{s}.jsonl" if s in ("hb", "wjb", "pap") else f"{TR}/classified_{s}.jsonl" for s in SETS}
LAYERS = json.load(open(f"{D}/acts/hb_meta.json"))["layers"]; TAIL = json.load(open(f"{D}/acts/hb_meta.json"))["tail_tokens"]
def keep(s):
    C = {r["pair_id"]: r for r in map(json.loads, open(CLS[s]))}; ids = json.load(open(f"{D}/acts/{s}_meta.json"))["pair_ids"]
    k = [i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    return np.array(k), np.array([int(C[ids[i]]["jailbroken"]) for i in k]), [C[ids[i]] for i in k]
K = {s: keep(s) for s in SETS}; hv = K["hb"][2]; yh = K["hb"][1]
def pool_split(seed, f=0.7):
    rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in hv}); tpl = sorted({r["jailbreak_idx"] for r in hv}); rng.shuffle(beh); rng.shuffle(tpl)
    trb, trt = set(beh[:int(len(beh) * f)]), set(tpl[:int(len(tpl) * f)]); teb, tet = set(beh[int(len(beh) * f):]), set(tpl[int(len(tpl) * f):])
    return (np.array([i for i, r in enumerate(hv) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]), np.array([i for i, r in enumerate(hv) if r["behavior_id"] in teb and r["jailbreak_idx"] in tet]))
splits = [(a, b) for a, b in (pool_split(s) for s in range(50)) if 0 < yh[b].sum() < len(b) and 0 < yh[a].sum() < len(a)][:20]
from sklearn.model_selection import StratifiedKFold
rnd = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yh)), yh))
names = json.load(open(f"{D}/in/trait_names_layer32.json"))

def run_layer(l):
    import torch
    from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
    mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    X = {s: np.load(f"{D}/acts/{s}_L{l}.npy", mmap_mode="r")[K[s][0]] for s in SETS}; Y = {s: K[s][1] for s in SETS}
    T = np.stack([torch.load(f"{D}/in/trait_vectors/{n}.pt", map_location="cpu")[l].float().numpy() for n in names]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    Gs = [np.random.default_rng(i).standard_normal(T.shape) for i in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
    out = {}
    for p in range(len(TAIL)):
        Z = {s: X[s][:, p].astype(np.float64) for s in SETS}; sc = {}; cell = {}
        for nm, V in [("PersonaJB", T)] + [(f"Gaussian{i}", g) for i, g in enumerate(Gs)] + [("fullprobe", None)]:
            F = {s: (Z[s] @ V.T if V is not None else Z[s]) for s in SETS}; C = 0.01 if V is None else 10.0; ns = 5 if V is None else len(splits)
            for pool in ("hb", "hbwjb"):
                fit = lambda idx: mk(C).fit(np.vstack([F["hb"][idx], F["wjb"]]) if pool == "hbwjb" else F["hb"][idx], np.r_[Y["hb"][idx], Y["wjb"]] if pool == "hbwjb" else Y["hb"][idx])
                dis, tra, save = [], {f: [] for f in FAM}, nm in ("PersonaJB", "Gaussian0", "fullprobe")
                for j, (tr, te) in enumerate(splits[:ns]):
                    m = fit(tr); s = m.decision_function(F["hb"]); dis.append(fold(Y["hb"][te], s[te]))
                    if save: sc[f"{nm}__{pool}__HB__s{j}"] = s.astype(np.float32)
                    for f in FAM:
                        sf = m.decision_function(F[f]); tra[f].append(fold(Y[f], sf))
                        if save: sc[f"{nm}__{pool}__{f}__s{j}"] = sf.astype(np.float32)
                oof = np.zeros(len(yh))
                for tr, te in rnd: oof[te] = fit(tr).decision_function(F["hb"][te])
                if save: sc[f"{nm}__{pool}__HB_randomCV_oof"] = oof.astype(np.float32)
                row = {"hb_randomCV": float(fold(Y["hb"], oof)), "hb_disjoint": float(np.mean(dis)), **{f: float(np.mean(v)) for f, v in tra.items()}}
                row["transfer_mean"] = float(np.mean([row[f] for f in FAM])); cell[f"{nm}__{pool}"] = row
        for pool in ("hb", "hbwjb"):
            cell[f"Gaussian__{pool}"] = {k: float(np.mean([cell[f"Gaussian{i}__{pool}"][k] for i in range(3)])) for k in cell[f"Gaussian0__{pool}"]}
        np.savez(f"{D}/scores/L{l}_p{p}.npz", **sc); out[f"L{l}_p{p}"] = {"layer": l, "pos": p - 9 if p < 9 else "user_mean", "token": TAIL[p], **cell}
        c = cell; print(f"L{l:2d} p{p} {TAIL[p]!r:14s} " + " | ".join(f"{pool}: P {c['PersonaJB__'+pool]['transfer_mean']:.3f} G {c['Gaussian__'+pool]['transfer_mean']:.3f} F {c['fullprobe__'+pool]['transfer_mean']:.3f} (HBcv P {c['PersonaJB__'+pool]['hb_randomCV']:.3f} G {c['Gaussian__'+pool]['hb_randomCV']:.3f})" for pool in ("hb", "hbwjb")), flush=True)
    return out

os.makedirs(f"{D}/scores", exist_ok=True)
print({s: (len(K[s][1]), int(K[s][1].sum())) for s in SETS}, "splits", len(splits), flush=True)
# sanity: the new L48 readout matches the transfer run's acts_pos_pez.pt
import torch
A = torch.load(f"{TR}/acts_pos_pez.pt"); old = A["X"][:, A["layers"].index(48)].float().numpy(); ids = json.load(open(f"{D}/acts/pez_meta.json"))["pair_ids"]
assert ids == A["pair_ids"]; new = np.load(f"{D}/acts/pez_L48.npy", mmap_mode="r")[:, :9].astype(np.float32)
cos = (old * new).sum(-1) / np.linalg.norm(old, axis=-1) / np.linalg.norm(new, axis=-1); print("sanity L48 pez cos old/new: min %.4f mean %.4f" % (cos.min(), cos.mean()), flush=True)
res = {}
for o in Parallel(n_jobs=NJ, backend="loky")(delayed(run_layer)(l) for l in LAYERS): res.update(o)
json.dump({"splits_used": len(splits), "tail_tokens": TAIL, "layers": LAYERS, "n": {s: [len(K[s][1]), int(K[s][1].sum())] for s in SETS}, "cells": res}, open(f"{D}/sweep.json", "w"), indent=1)
print("done", flush=True)
