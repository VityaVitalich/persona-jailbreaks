"""PersonaJB vs Gaussian control vs raw probe on DOMINIC's OLMo-3-7B detection activations (RCP full_trait_output/
{harmbench,wildjailbreak}_activations_olmo3, {pair,pap,gptfuzz,pez}_activations_olmo3_hb; L16 pre-generation last token).
His GCG run never finished (his sweep's GCG column read the Llama gcg_activations by default), so no GCG here.
Same protocol as persona_four.py: C tuned on a behaviour-grouped val split of the training set (HB also template-disjoint),
transfer = mean folded AUC over the other attack families (pair, pap, gptfuzz, pez); behaviour-cluster bootstrap B=1000,
seed 12345; paired deltas. Also: cosine between his and our HB activations for shared pair_ids.
Saves every per-prompt score to scores_dom_olmo_compare.npz, table dom_olmo_compare.json."""
import json, glob, os, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
sys.path.insert(0, ".."); import olmo_probe_lib as L
DD = sys.argv[1]   # local copy of Dominic's dirs
DIRS = {"hb": "harmbench_activations_olmo3", "wjb": "wildjailbreak_activations_olmo3", "pair": "pair_activations_olmo3_hb",
        "pap": "pap_activations_olmo3_hb", "gptfuzz": "gptfuzz_activations_olmo3_hb", "pez": "pez_activations_olmo3_hb"}
SETS = list(DIRS); FAM = ["pair", "pap", "gptfuzz", "pez"]; G = 3; B = 1000
def dom(st):
    d = f"{DD}/{DIRS[st]}"
    R = [json.loads(l) for l in open(f"{d}/classified_responses.jsonl")]
    A = torch.load(f"{d}/activations.pt", map_location="cpu", weights_only=False)
    R = [r for r in R if r.get("jailbroken") is not None and r["pair_id"] in A and (st != "hb" or r["attack_type"] == "human_jailbreak")]
    return np.stack([A[r["pair_id"]]["16"].float().numpy() for r in R]).astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R
D = {s: dom(s) for s in SETS}
print({s: (len(D[s][1]), int(D[s][1].sum())) for s in SETS}, flush=True)
T, names = L.trait_matrix("olmo"); d = T.shape[1]
mats = {"PersonaJB": T}
for g in range(G):
    Z = np.random.default_rng(1000 + g).standard_normal((len(T), d)); mats[f"Gaussian{g}"] = Z / np.linalg.norm(Z, axis=1, keepdims=True)
mats["raw"] = None
f = lambda X, M: X if M is None else X @ M.T
jobs = [delayed(L.cell)((det, a), a, f(D[a][0], M), D[a][1], D[a][2], {b: f(D[b][0], M) for b in FAM if b != a}) for det, M in mats.items() for a in SETS]
OUT = dict(Parallel(n_jobs=8, backend="loky")(jobs)); print("fits done", flush=True)
SC = {}; res = {"sets": SETS, "fam": FAM, "n": {s: [len(D[s][1]), int(D[s][1].sum())] for s in SETS}, "per_family": {}, "C": {}, "cells": {}}
for ((det, a), _), v in OUT.items():
    p = f"{det}__train_{a}"; SC[f"{p}__val_idx"] = v["val_idx"]; res["C"][p] = v["C"]
    for C, s in v["val_scores"].items(): SC[f"{p}__val_C{C:g}"] = s
    for b, s in v["test"].items(): SC[f"{p}__test_{b}__C{v['C']:g}"] = s
behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {b: i for i, b in enumerate(behs)}
W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)
bidx = {s: np.array([bi[r["behavior_id"]] for r in D[s][2]]) for s in SETS}
def boot(y, s, bo):
    sg = 1 if roc_auc_score(y, s) >= 0.5 else -1
    return roc_auc_score(y, sg * s), np.array([roc_auc_score(y, sg * s, sample_weight=W[k][bo]) if (W[k][bo] * y).sum() > 0 and (W[k][bo] * (1 - y)).sum() > 0 else np.nan for k in range(B)])
TT = {}
for det in mats:
    for a in SETS:
        fams = [x for x in FAM if x != a]; pts, bss = zip(*[boot(D[x][1], OUT[((det, a), a)]["test"][x], bidx[x]) for x in fams])
        TT[(det, a)] = (np.mean(pts), np.nanmean(np.stack(bss), 0)); res["per_family"][f"{det}__{a}"] = dict(zip(fams, map(float, pts)))
    TT[(det, "avg")] = (np.mean([TT[(det, a)][0] for a in SETS]), np.mean([TT[(det, a)][1] for a in SETS], 0))
ci = lambda b: [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
for a in SETS + ["avg"]:
    TT[("Gaussian", a)] = (np.mean([TT[(f"Gaussian{g}", a)][0] for g in range(G)]), np.mean([TT[(f"Gaussian{g}", a)][1] for g in range(G)], 0))
    c = {det: float(TT[(det, a)][0]) for det in ("PersonaJB", "Gaussian", "raw")}
    for o in ("Gaussian", "raw"): c[f"PersonaJB-{o}"] = c["PersonaJB"] - c[o]; c[f"PersonaJB-{o}_ci95"] = ci(TT[("PersonaJB", a)][1] - TT[(o, a)][1])
    for det in ("PersonaJB", "Gaussian", "raw"): SC[f"boot__{det}__{a}"] = TT[(det, a)][1].astype(np.float32)
    res["cells"][a] = c
for s in SETS: SC[f"labels_{s}"] = D[s][1]; SC[f"pair_ids_{s}"] = np.array([str(r["pair_id"]) for r in D[s][2]]); SC[f"behavior_ids_{s}"] = np.array([r["behavior_id"] for r in D[s][2]])
# his vs our HB activations for shared pair_ids (L16, last token)
Xo, yo, Ro = L.olmo("hb"); oi = {r["pair_id"]: i for i, r in enumerate(Ro)}
sh = [(i, oi[r["pair_id"]]) for i, r in enumerate(D["hb"][2]) if r["pair_id"] in oi]
if sh:
    a_, b_ = D["hb"][0][[i for i, _ in sh]], Xo[[j for _, j in sh]]
    cs = (a_ * b_).sum(1) / np.linalg.norm(a_, axis=1) / np.linalg.norm(b_, axis=1)
    res["hb_act_cos_vs_ours"] = {"n_shared": len(sh), "median": float(np.median(cs)), "p1": float(np.percentile(cs, 1)), "min": float(cs.min())}
    SC["hb_act_cos_vs_ours"] = cs.astype(np.float32)
json.dump(res, open("dom_olmo_compare.json", "w"), indent=1); np.savez("scores_dom_olmo_compare.npz", **SC)
print("hb act cos vs ours:", res.get("hb_act_cos_vs_ours"))
for det in ("PersonaJB", "Gaussian", "raw"): print(f"{det:10s} " + " ".join(f"{a}:{res['cells'][a][det]:.3f}" for a in SETS + ["avg"]))
for o in ("Gaussian", "raw"): print(f"PJB-{o:8s} " + " ".join(f"{a}:{res['cells'][a][f'PersonaJB-{o}']:+.3f}" for a in SETS + ["avg"]), "avg CI", res["cells"]["avg"][f"PersonaJB-{o}_ci95"])
