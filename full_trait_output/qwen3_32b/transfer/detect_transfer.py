"""Transfer to Qwen's own GCG/PAIR/PEZ/GPTFuzz test cases + PAP: PersonaJB vs Gaussian vs full probe, Qwen3-32B layer 48,
per chat-template tail position (detect_pos.py protocol, unchanged training).
Same protocol as stage2/qwen_detect.py (StandardScaler + LR C=10 on 240 projections; HB human_jailbreak train pool + all
WJB; 50 pool splits with both classes; folded AUC on HB test and each transfer family). Full probe: LR C=0.01 on 5120 dims, 10 splits.
Saves per-prompt scores (every split, set, position; PersonaJB, Gaussian draw 0, full probe) to scores_transfer.npz; table to detect_transfer.json."""
import os
import json, random, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
LAYER = 48; Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"
def load(st):
    own = st not in ("hb", "wjb", "pap")
    A = torch.load(f"acts_pos_{st}.pt" if own else f"{Q}/l48_run/acts_pos_{st}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"classified_{st}.jsonl" if own else f"{Q}/stage2_out/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    return A["X"][k, A["layers"].index(LAYER)].float().numpy().astype(np.float64), np.array([int(C[A["pair_ids"][i]]["jailbroken"]) for i in k]), [C[A["pair_ids"][i]] for i in k], A["tail_tokens"]
Xh, yh, hv, tail = load("hb"); Xa, ya, _, _ = load("wjb")
FAM = [f for f in ("gcg", "pair", "pap", "gptfuzz", "pez") if f == "pap" or os.path.exists(f"classified_{f}.jsonl")]
L = {f: load(f) for f in FAM}; E = {f: v[:2] for f, v in L.items()}; print({f: (len(y), int(y.sum())) for f, (_, y) in E.items()}, flush=True)
names = json.load(open(f"{Q}/trait_names_layer32.json"))
T = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[LAYER].float().numpy() for n in names]).astype(np.float64)
T /= np.linalg.norm(T, axis=1, keepdims=True)
Gs = [np.random.default_rng(i).standard_normal(T.shape) for i in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
def pool_split(seed, f=0.7):
    rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in hv}); tpl = sorted({r["jailbreak_idx"] for r in hv}); rng.shuffle(beh); rng.shuffle(tpl)
    trb, trt = set(beh[:int(len(beh) * f)]), set(tpl[:int(len(tpl) * f)]); teb, tet = set(beh[int(len(beh) * f):]), set(tpl[int(len(tpl) * f):])
    return (np.array([i for i, r in enumerate(hv) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]),
            np.array([i for i, r in enumerate(hv) if r["behavior_id"] in teb and r["jailbreak_idx"] in tet]))
splits = [(a, b) for a, b in (pool_split(s) for s in range(50)) if 0 < yh[b].sum() < len(b) and 0 < yh[a].sum() < len(a)]
mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))
res, sc = {"tail_tokens": tail, "splits_used": len(splits), "families": {f: [len(y), int(y.sum())] for f, (_, y) in E.items()}, "positions": {}}, {"labels_HB": yh, **{f"labels_{f}": y for f, (_, y) in E.items()}, **{f"pair_ids_{f}": np.array([r["pair_id"] for r in v[2]]) for f, v in L.items()}}
for p in range(Xh.shape[1]):
    pos = p - Xh.shape[1]; h, a = Xh[:, p], Xa[:, p]; row = {"token": tail[p]}
    def run(nm, feat, C, sp_list, save):
        hb, tf = [], {f: [] for f in FAM}
        for j, (tr, te) in enumerate(sp_list):
            m = mk(C).fit(np.vstack([feat(h[tr]), feat(a)]), np.r_[yh[tr], ya]); s = m.decision_function(feat(h)); hb.append(fold(yh[te], s[te]))
            if save: sc[f"{nm}__pos{pos}__HB__s{j}"] = s.astype(np.float32)
            for f, (X, y) in E.items():
                sf = m.decision_function(feat(X[:, p])); tf[f].append(fold(y, sf))
                if save: sc[f"{nm}__pos{pos}__{f}__s{j}"] = sf.astype(np.float32)
        r = {"hb": float(np.mean(hb)), **{f: float(np.mean(v)) for f, v in tf.items()}}; r["transfer_mean"] = float(np.mean([r[f] for f in FAM])); return r
    for nm, V in [("PersonaJB", T)] + [(f"Gaussian{i}", g) for i, g in enumerate(Gs)]:
        row[nm] = run(nm, lambda X, V=V: X @ V.T, 10.0, splits, nm in ("PersonaJB", "Gaussian0"))
    row["Gaussian"] = {k: float(np.mean([row[f"Gaussian{i}"][k] for i in range(3)])) for k in row["Gaussian0"]}
    row["full probe"] = run("fullprobe", lambda X: X, 0.01, splits[:10], True)
    res["positions"][str(pos)] = row
    print(f"pos {pos:3d} {tail[p]!r:14s} " + " | ".join(f"{nm} " + " ".join(f"{k} {row[nm][k]:.3f}" for k in ["hb"] + FAM + ["transfer_mean"]) for nm in ("PersonaJB", "Gaussian", "full probe")), flush=True)
for j, (tr, te) in enumerate(splits): sc[f"idx_train__s{j}"] = tr; sc[f"idx_test__s{j}"] = te
np.savez("scores_transfer.npz", pair_ids_HB=np.array([r["pair_id"] for r in hv]), **sc); json.dump(res, open("detect_transfer.json", "w"), indent=1)
