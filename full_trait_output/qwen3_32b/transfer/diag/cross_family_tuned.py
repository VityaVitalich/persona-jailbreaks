"""Cross-family transfer with C tuned on a validation split of the TRAINING set (never on test/transfer sets).
Models: Qwen3-32B (L48, last prompt token) and Llama-3.1-8B (L16, shared data). Detectors: PersonaJB (unit trait vectors),
Gaussian (3 draws of unit random directions, as many as traits), full probe; all StandardScaler + LR with the same C grid.
Per training set: 80/20 split grouped by behaviour (HB: behaviour- AND template-disjoint, seeds until both classes present);
C = argmax folded val AUC; refit on the whole training set; score every other set.
Uncertainty: cluster bootstrap over HarmBench behaviours (B=1000, shared across all test sets), AUC orientation fixed at the
point estimate. Per cell: transfer = mean AUC over GCG/PAIR/PAP/GPTFuzz/PEZ minus the training set. Paired deltas
PersonaJB-Gaussian(mean of draws) and PersonaJB-full, Bonferroni over the 8 columns x 2 models.
Saves val scores for every C, test scores for the chosen C, bootstrap draws: scores_cross_family_tuned.npz; table json."""
import json, random, sys, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"; TR = f"{Q}/transfer"; LD = f"{Q}/llama_control/data"
SETS = ["hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"]; FAM = ["gcg", "pair", "pap", "gptfuzz", "pez"]
GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; B = 1000
def qwen(st):
    own = st not in ("hb", "wjb", "pap")
    A = torch.load(f"{TR}/acts_pos_{st}.pt" if own else f"{Q}/l48_run/acts_pos_{st}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"{TR}/classified_{st}.jsonl" if own else f"{Q}/stage2_out/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    R = [C[A["pair_ids"][i]] for i in k]
    return A["X"][k, A["layers"].index(48), -1].float().numpy().astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R
def llama(st):
    d = {"hb": "harmbench", "wjb": "wildjailbreak"}.get(st, st)
    try: R = [json.loads(l) for l in open(f"{LD}/{d}/classified_responses.jsonl")]
    except FileNotFoundError: R = [json.loads(l) for l in open(f"{LD}/{d}/responses.jsonl")]
    A = torch.load(f"{LD}/{d}/activations.pt", map_location="cpu", weights_only=False)
    R = [r for r in R if r.get("jailbroken") is not None and r["pair_id"] in A and (st != "hb" or r["attack_type"] == "human_jailbreak")]
    return np.stack([A[r["pair_id"]]["16"].float().numpy() for r in R]).astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
def val_split(R, y, hb, seed0=0):
    for seed in range(seed0, seed0 + 200):
        rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in R}); rng.shuffle(beh); vb = set(beh[:int(0.2 * len(beh))])
        if hb:
            tp = sorted({r["jailbreak_idx"] for r in R}); rng.shuffle(tp); vt = set(tp[:int(0.2 * len(tp))])
            va = np.array([i for i, r in enumerate(R) if r["behavior_id"] in vb and r["jailbreak_idx"] in vt]); tr = np.array([i for i, r in enumerate(R) if r["behavior_id"] not in vb and r["jailbreak_idx"] not in vt])
        else:
            va = np.array([i for i, r in enumerate(R) if r["behavior_id"] in vb]); tr = np.array([i for i, r in enumerate(R) if r["behavior_id"] not in vb])
        if 0 < y[va].sum() < len(va) and 0 < y[tr].sum() < len(tr): return tr, va
    raise RuntimeError("no valid val split")
def cell(model, det, a, Fa, ya, Ra, tests):
    tr, va = val_split(Ra, ya, a == "hb"); vs = {}; vauc = {}
    for C in GRID:
        s = mk(C).fit(Fa[tr], ya[tr]).decision_function(Fa[va]); vs[C] = s.astype(np.float32); vauc[C] = fold(ya[va], s)
    Cb = max(GRID, key=lambda c: vauc[c]); m = mk(Cb).fit(Fa, ya)
    return (model, det, a), {"C": Cb, "val_auc": vauc, "val_idx": va, "val_scores": vs, "test": {b: m.decision_function(Fb).astype(np.float32) for b, Fb in tests.items()}}
OUT, SC, TAB = {}, {}, {}
rng_b = np.random.default_rng(12345)
for model, loader, Tpath in (("qwen", qwen, None), ("llama", llama, f"{LD}/trait_matrix_layer16.npy")):
    D = {s: loader(s) for s in SETS}
    if Tpath is None:
        names = json.load(open(f"{Q}/trait_names_layer32.json")); T = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[48].float().numpy() for n in names]).astype(np.float64)
    else: T = np.load(Tpath).astype(np.float64)
    T /= np.linalg.norm(T, axis=1, keepdims=True); Gs = [np.random.default_rng(g).standard_normal(T.shape) for g in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
    dets = [("PersonaJB", T)] + [(f"Gaussian{g}", G) for g, G in enumerate(Gs)] + [("fullprobe", None)]
    jobs = []
    for det, V in dets:
        F = {s: (D[s][0] @ V.T if V is not None else D[s][0]) for s in SETS}
        for a in SETS: jobs.append(delayed(cell)(model, det, a, F[a], D[a][1], D[a][2], {b: F[b] for b in SETS if b != a}))
    for k, v in Parallel(n_jobs=8, backend="loky")(jobs): OUT[k] = v
    print(model, "fits done", flush=True)
    # cluster bootstrap over behaviours, shared across test sets
    behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {b: i for i, b in enumerate(behs)}
    W = rng_b.multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)   # [B, n_beh] resample counts
    bidx = {s: np.array([bi[r["behavior_id"]] for r in D[s][2]]) for s in SETS}
    def auc_boot(y, s, b_of):
        sign = 1 if roc_auc_score(y, s) >= 0.5 else -1; pt = roc_auc_score(y, sign * s)
        bs = np.array([roc_auc_score(y, sign * s, sample_weight=W[k][b_of]) if (W[k][b_of] * y).sum() > 0 and (W[k][b_of] * (1 - y)).sum() > 0 else np.nan for k in range(B)])
        return pt, bs
    for det, _ in dets:
        for a in SETS:
            fams = [f for f in FAM if f != a]; pts, bss = [], []
            for f in fams:
                pt, bs = auc_boot(D[f][1], OUT[(model, det, a)]["test"][f], bidx[f]); pts.append(pt); bss.append(bs)
            TAB[(model, det, a)] = (np.mean(pts), np.nanmean(np.stack(bss), 0), dict(zip(fams, pts)))
    for s in SETS: SC[f"{model}__labels_{s}"] = D[s][1]; SC[f"{model}__behavior_ids_{s}"] = np.array([r["behavior_id"] for r in D[s][2]])
for (model, det, a), v in OUT.items():
    p = f"{model}__{det}__train_{a}"; SC[f"{p}__val_idx"] = v["val_idx"]
    for C, s in v["val_scores"].items(): SC[f"{p}__val_C{C:g}"] = s
    for b, s in v["test"].items(): SC[f"{p}__test_{b}__C{v['C']:g}"] = s
for k, (pt, bs, per) in TAB.items(): SC["boot__" + "__".join(k)] = bs.astype(np.float32)
# assemble table
NB = 16; zq = 1 - 0.05 / NB / 2
res = {}
for model in ("llama", "qwen"):
    cols = SETS + ["avg"]
    get = lambda det, a: (TAB[(model, det, a)][0], TAB[(model, det, a)][1]) if a != "avg" else (np.mean([TAB[(model, det, x)][0] for x in SETS]), np.mean([TAB[(model, det, x)][1] for x in SETS], 0))
    for det in ("PersonaJB", "Gaussian", "fullprobe"):
        for a in cols:
            if det == "Gaussian": pt = np.mean([get(f"Gaussian{g}", a)[0] for g in range(3)]); bs = np.mean([get(f"Gaussian{g}", a)[1] for g in range(3)], 0)
            else: pt, bs = get(det, a)
            Cs = None if a == "avg" else ([OUT[(model, f"Gaussian{g}", a)]["C"] for g in range(3)] if det == "Gaussian" else OUT[(model, det, a)]["C"])
            res[f"{model}__{det}__{a}"] = {"auc": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "C": Cs}
        if det != "PersonaJB":
            for a in cols:
                pj = get("PersonaJB", a)
                ot = (np.mean([get(f"Gaussian{g}", a)[0] for g in range(3)]), np.mean([get(f"Gaussian{g}", a)[1] for g in range(3)], 0)) if det == "Gaussian" else get(det, a)
                d = pj[1] - ot[1]; res[f"{model}__delta_PersonaJB_minus_{det}__{a}"] = {"delta": float(pj[0] - ot[0]), "ci_bonf": [float(np.nanpercentile(d, 100 * (1 - zq))), float(np.nanpercentile(d, 100 * zq))]}
json.dump({"grid": GRID, "B": B, "bonferroni_m": NB, "cells": res, "per_family": {"__".join(k): v[2] for k, v in TAB.items()}}, open("cross_family_tuned.json", "w"), indent=1)
np.savez("scores_cross_family_tuned.npz", **SC)
for model in ("llama", "qwen"):
    for det in ("PersonaJB", "Gaussian", "fullprobe"):
        print(f"{model:5s} {det:9s} " + " | ".join(f"{a}: {res[f'{model}__{det}__{a}']['auc']:.3f} [{res[f'{model}__{det}__{a}']['ci95'][0]:.3f},{res[f'{model}__{det}__{a}']['ci95'][1]:.3f}] C={res[f'{model}__{det}__{a}']['C']}" for a in SETS + ["avg"]))
    for det in ("Gaussian", "fullprobe"):
        print(f"{model:5s} PJB-{det:9s} " + " | ".join(f"{a}: {res[f'{model}__delta_PersonaJB_minus_{det}__{a}']['delta']:+.3f} [{res[f'{model}__delta_PersonaJB_minus_{det}__{a}']['ci_bonf'][0]:+.3f},{res[f'{model}__delta_PersonaJB_minus_{det}__{a}']['ci_bonf'][1]:+.3f}]" for a in SETS + ["avg"]))
