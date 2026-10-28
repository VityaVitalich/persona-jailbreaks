#!/usr/bin/env python3
"""Qwen3-32B detection: PersonaJB and the cheap non-steering controls, layer 32, last prompt token.
Detector (Gemma protocol of trait_pc_sweep.py): StandardScaler + LR C=10 fit on the HarmBench train pool plus all 2000
WildJailbreak rows; 50 HarmBench pool splits (S.pool_split); folded AUC on the HarmBench test pool (splits with both
classes) and on PAP (the only transfer family with target-agnostic test cases so far).
Dictionaries (K = 240 unit rows): trait vectors (PersonaJB); Alpaca / No Robots / Wikipedia activations, shuffled-label
contrasts, random Gaussian (draws 0-2, seeds as build_bases_gemma.py); each also with the persona component removed and
with a random 240-dim span removed. Single directions: dmu (mean jailbroken - mean refused, HB train rows of each
split), refusal-like (mean plain harmful - mean plain Alpaca, role-play set; NOT the Arditi recipe), each full / persona
component / persona removed; random-direction null (1000).
Saves every per-prompt score (split x set) for PersonaJB to scores_personajb.npz; summary to qwen_detection.json."""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from joblib import Parallel, delayed
Q = Path(__file__).resolve().parent.parent; O = Q.parent; sys.path.insert(0, str(O / "trait_pc_sweep")); sys.path.insert(0, str(O / "norm_share_controls"))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402
D = Q / "data"; LAYER = 30 + 2
fold = lambda a: np.maximum(a, 1 - a); unit = lambda v: v / np.linalg.norm(v); un = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
def load(s):
    rows = S.load_jsonl(D / f"classified_{s}.jsonl"); rows = [r for r in rows if r["jailbroken"] is not None]
    return S.build_matrix(rows, torch.load(D / f"acts_{s}.pt", map_location="cpu", weights_only=False), LAYER)
Xh, yh, hv = load("hb"); keep = [i for i, r in enumerate(hv) if r["attack_type"] == "human_jailbreak"]
Xh, yh, hv = Xh[keep].astype(np.float64), yh[keep], [hv[i] for i in keep]
Xa, ya, _ = load("wjb"); Xa = Xa.astype(np.float64)
fam = {"PAP": tuple(a.astype(np.float64) if i == 0 else a for i, a in enumerate(load("pap")[:2]))}
T = un(np.load(Q / "trait_matrix_layer32.npy").astype(np.float64)); K = len(T); d = T.shape[1]; Qp = np.linalg.qr(T.T)[0]
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]; splits = [(a, b) for a, b in splits if 0 < yh[b].sum() < len(b)]
print(f"HB {len(yh)} jb {yh.sum()} | WJB {len(ya)} jb {ya.sum()} | PAP {len(fam['PAP'][1])} jb {fam['PAP'][1].sum()} | {len(splits)} splits", flush=True)
def detector(V, save=None):
    hbs, xfs, sc = [], [], {}
    for k, (tr, te) in enumerate(splits):
        m = S.model(0).fit(np.vstack([Xh[tr] @ V.T, Xa @ V.T]), np.r_[yh[tr], ya])
        sh = m.decision_function(Xh @ V.T); hbs.append(fold(rank_auc(yh[te], sh[te][:, None]))[0])
        sf = {f: m.decision_function(X @ V.T) for f, (X, y) in fam.items()}
        xfs.append(np.mean([fold(rank_auc(fam[f][1], s[:, None]))[0] for f, s in sf.items()]))
        if save is not None: sc[f"HB__s{k}"] = sh; sc.update({f"{f}__s{k}": s for f, s in sf.items()}); sc[f"idx_test__s{k}"] = te; sc[f"idx_train__s{k}"] = tr
    if save: np.savez(save, labels_HB=yh, **{f"labels_{f}": y for f, (X, y) in fam.items()}, **sc)
    return float(np.mean(hbs)), float(np.mean(xfs))
res = {"n": dict(hb=len(yh), hb_jb=int(yh.sum()), wjb=len(ya), wjb_jb=int(ya.sum()), pap=len(fam["PAP"][1]), pap_jb=int(fam["PAP"][1].sum()), splits=len(splits))}
res["PersonaJB"] = detector(T, save=Q / "analysis/scores_personajb.npz"); print("PersonaJB HB / PAP", np.round(res["PersonaJB"], 3), flush=True)
ex = torch.load(D / "extra_acts.pt", map_location="cpu") if (D / "extra_acts.pt").exists() else None
if ex is not None:
    rp = [json.loads(l) for l in open(O / "roleplay_check/synthetic_v2_prompts.jsonl")]
    A = np.stack([ex["rp:" + r["id"]].float().numpy() for r in rp if r["source"] == "alpaca" and r["frame"] == "plain"]).astype(np.float64)
    Hm = np.stack([ex["rp:" + r["id"]].float().numpy() for r in rp if r["harm"] == "harmful" and r["frame"] == "plain"]).astype(np.float64)
    W = lambda p: np.stack([ex[k].float().numpy() for k in sorted((k for k in ex if k.startswith(p)), key=lambda s: int(s.split("_")[1]))]).astype(np.float64)
    Wk, NR = W("wiki_"), W("nr_")
    rng = np.random.default_rng(0); B = {}
    for i in range(3):
        B[f"Alpaca acts|{i}"] = un(A[rng.choice(len(A), K, replace=False)])
        B[f"Shuffled contrasts|{i}"] = un(np.stack([(lambda p: A[p[:len(A)//2]].mean(0) - A[p[len(A)//2:]].mean(0))(rng.permutation(len(A))) for _ in range(K)]))
        B[f"Random Gaussian|{i}"] = un(rng.standard_normal((K, d)))
    r1 = np.random.default_rng(1)
    for n, M in (("Wikipedia acts", Wk), ("No Robots acts", NR)):
        for i in range(3): B[f"{n}|{i}"] = un(M[r1.choice(len(M), K, replace=False)])
    Qr = np.linalg.qr(np.random.default_rng(4).standard_normal((K, d)).T)[0]
    def run(n, V):
        return n, float(np.mean(np.sum((V @ Qp) ** 2, 1))), detector(V), detector(un(V - V @ Qp @ Qp.T)), detector(un(V - V @ Qr @ Qr.T))
    out = Parallel(n_jobs=8)(delayed(run)(n, V) for n, V in B.items())
    res["dicts"] = {n: dict(share=s, whole=w, persona_removed=p, random_removed=r) for n, s, w, p, r in out}
    print(f"{'dictionary':22s} share | whole HB PAP | persona removed | random span removed")
    for n in dict.fromkeys(k.split("|")[0] for k in B):
        v = [res["dicts"][k] for k in res["dicts"] if k.split("|")[0] == n]; m = lambda key: np.mean([x[key] for x in v], 0)
        print(f"{n:22s} {np.mean([x['share'] for x in v]):.3f} | {m('whole')[0]:.3f} {m('whole')[1]:.3f} | {m('persona_removed')[0]:.3f} {m('persona_removed')[1]:.3f} | {m('random_removed')[0]:.3f} {m('random_removed')[1]:.3f}", flush=True)
    Gn = un(np.random.default_rng(0).standard_normal((1000, d)))
    null = {"HB": fold(rank_auc(yh, Xh @ Gn.T)), "PAP": fold(rank_auc(fam["PAP"][1], fam["PAP"][0] @ Gn.T))}
    res["random_direction_null"] = {k: dict(mean=float(v.mean()), p95=float(np.percentile(v, 95))) for k, v in null.items()}
    ref = unit(Hm.mean(0) - A.mean(0)); dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))
    res["directions"] = {}
    print(f"\nrandom direction null: HB mean {null['HB'].mean():.3f} (95th {np.percentile(null['HB'], 95):.3f}), PAP mean {null['PAP'].mean():.3f} (95th {np.percentile(null['PAP'], 95):.3f})")
    for n, v in (("refusal-like (harmful - Alpaca plain)", ref), ("dmu (all HB rows; HB in-sample)", dmu)):
        pin = Qp @ (Qp.T @ v)
        for part, w in (("full", v), ("persona component", unit(pin)), ("persona removed", unit(v - pin))):
            a = fold(rank_auc(yh, (Xh @ w)[:, None]))[0]; x = fold(rank_auc(fam["PAP"][1], (fam["PAP"][0] @ w)[:, None]))[0]
            res["directions"][f"{n}|{part}"] = dict(share=float(pin @ pin), hb=float(a), pap=float(x), pap_pct_null=float(np.mean(null["PAP"] < x)))
            print(f"  {n:38s} {part:18s} share {pin @ pin:.3f}  HB {a:.3f}  PAP {x:.3f} (pct {100*np.mean(null['PAP'] < x):.0f})")
json.dump(res, open(Q / "analysis/qwen_detection.json", "w"), indent=1)
