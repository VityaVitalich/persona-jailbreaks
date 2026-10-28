"""Within-family probe AUC with C tuned on training folds only (nested): 5-fold GroupKFold by behaviour; inside each
training fold an 80/20 behaviour-grouped val split picks C from the grid; refit on the training fold; score the test fold.
Qwen3-32B L48 last token, Llama-3.1-8B L16; LR raw / PersonaJB / Gaussian (3 draws). CI: behaviour-clustered bootstrap
(B=1000) of the out-of-fold AUC. OOF scores -> scores_within_family_tuned.npz, table -> within_family_tuned.json."""
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

import sys
from sklearn.model_selection import GroupKFold
def within(model, det, s, F, y, R):
    g = np.array([r["behavior_id"] for r in R]); oof = np.zeros(len(y)); cs = []
    for tr, te in GroupKFold(5).split(F, y, g):
        Rtr = [R[i] for i in tr]; itr, iva = val_split(Rtr, y[tr], False)
        vauc = {C: fold(y[tr][iva], mk(C).fit(F[tr][itr], y[tr][itr]).decision_function(F[tr][iva])) for C in GRID}
        Cb = max(GRID, key=lambda c: vauc[c]); cs.append(Cb); oof[te] = mk(Cb).fit(F[tr], y[tr]).decision_function(F[te])
    return (model, det, s), (oof, cs)
res, SC = {}, {}
for model, loader, Tpath in (("qwen", qwen, None), ("llama", llama, f"{LD}/trait_matrix_layer16.npy")):
    D = {s: loader(s) for s in SETS}
    if Tpath is None:
        names = json.load(open(f"{Q}/trait_names_layer32.json")); T = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[48].float().numpy() for n in names]).astype(np.float64)
    else: T = np.load(Tpath).astype(np.float64)
    T /= np.linalg.norm(T, axis=1, keepdims=True); Gs = [np.random.default_rng(g).standard_normal(T.shape) for g in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
    dets = [("LR_raw", None), ("LR_PersonaJB", T)] + [(f"LR_Gaussian{g}", G) for g, G in enumerate(Gs)]
    out = dict(Parallel(n_jobs=6)(delayed(within)(model, d, s, D[s][0] @ V.T if V is not None else D[s][0], D[s][1], D[s][2]) for d, V in dets for s in SETS))
    rng = np.random.default_rng(12345)
    for s in SETS:
        y = D[s][1]; g = np.array([r["behavior_id"] for r in D[s][2]]); ub, inv = np.unique(g, return_inverse=True)
        W = rng.multinomial(len(ub), np.ones(len(ub)) / len(ub), size=1000)[:, inv]
        SC[f"{model}__labels_{s}"] = y; SC[f"{model}__behavior_ids_{s}"] = g
        for d, _ in dets:
            oof, cs = out[(model, d, s)]; sg = 1 if roc_auc_score(y, oof) >= 0.5 else -1
            bs = np.array([roc_auc_score(y, sg * oof, sample_weight=W[k]) if (W[k] * y).sum() > 0 and (W[k] * (1 - y)).sum() > 0 else np.nan for k in range(1000)])
            res[f"{model}__{d}__{s}"] = {"auc": float(roc_auc_score(y, sg * oof)), "boot": bs, "C_per_fold": cs}; SC[f"{model}__{d}__{s}__oof"] = oof.astype(np.float32)
        for k in ("auc",):
            pt = np.mean([res[f"{model}__LR_Gaussian{i}__{s}"]["auc"] for i in range(3)]); bs = np.nanmean([res[f"{model}__LR_Gaussian{i}__{s}"]["boot"] for i in range(3)], 0)
            res[f"{model}__LR_Gaussian__{s}"] = {"auc": float(pt), "boot": bs}
    for d in ("LR_raw", "LR_PersonaJB", "LR_Gaussian"):
        print(f"{model:5s} {d:13s} " + " | ".join(f"{s}: {res[f'{model}__{d}__{s}']['auc']:.3f} [{np.nanpercentile(res[f'{model}__{d}__{s}']['boot'], 2.5):.3f},{np.nanpercentile(res[f'{model}__{d}__{s}']['boot'], 97.5):.3f}]" for s in SETS), flush=True)
for k, v in res.items():
    b = v.pop("boot"); v["ci95"] = [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]; SC[f"boot__{k}"] = b.astype(np.float32)
json.dump(res, open("within_family_tuned.json", "w"), indent=1); np.savez("scores_within_family_tuned.npz", **SC)
