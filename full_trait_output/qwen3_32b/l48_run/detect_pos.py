"""PersonaJB vs Gaussian vs full-activation probe on Qwen3-32B layer 48, per chat-template tail position.
Same protocol as stage2/qwen_detect.py (StandardScaler + LR C=10 on 240 projections; HB human_jailbreak train pool + all
WJB; 50 pool splits with both classes; folded AUC on HB test and PAP). Full probe: LR C=0.01 on 5120 dims, 10 splits.
Saves per-prompt scores (every split, set, position; PersonaJB, Gaussian draw 0) to scores_pos.npz; table to detect_pos.json."""
import json, random, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
LAYER = 48; Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"
def load(st):
    A = torch.load(f"acts_pos_{st}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"{Q}/stage2_out/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    return A["X"][k, A["layers"].index(LAYER)].float().numpy().astype(np.float64), np.array([int(C[A["pair_ids"][i]]["jailbroken"]) for i in k]), [C[A["pair_ids"][i]] for i in k], A["tail_tokens"]
Xh, yh, hv, tail = load("hb"); Xa, ya, _, _ = load("wjb"); Xp, yp, _, _ = load("pap")
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
res, sc = {"tail_tokens": tail, "splits_used": len(splits), "positions": {}}, {"labels_HB": yh, "labels_PAP": yp}
for p in range(Xh.shape[1]):
    pos = p - Xh.shape[1]; h, a, q = Xh[:, p], Xa[:, p], Xp[:, p]; row = {"token": tail[p]}
    for nm, V in [("PersonaJB", T)] + [(f"Gaussian{i}", g) for i, g in enumerate(Gs)]:
        hb, pap = [], []
        for j, (tr, te) in enumerate(splits):
            m = mk(10.0).fit(np.vstack([h[tr] @ V.T, a @ V.T]), np.r_[yh[tr], ya]); s, sp = m.decision_function(h @ V.T), m.decision_function(q @ V.T)
            hb.append(fold(yh[te], s[te])); pap.append(fold(yp, sp))
            if nm in ("PersonaJB", "Gaussian0"): sc[f"{nm}__pos{pos}__HB__s{j}"] = s.astype(np.float32); sc[f"{nm}__pos{pos}__PAP__s{j}"] = sp.astype(np.float32)
        row[nm] = {"hb": float(np.mean(hb)), "pap": float(np.mean(pap))}
    row["Gaussian"] = {k: float(np.mean([row[f"Gaussian{i}"][k] for i in range(3)])) for k in ("hb", "pap")}
    hb, pap = [], []
    for j, (tr, te) in enumerate(splits[:10]):
        m = mk(0.01).fit(np.vstack([h[tr], a]), np.r_[yh[tr], ya]); s, sp = m.decision_function(h), m.decision_function(q)
        hb.append(fold(yh[te], s[te])); pap.append(fold(yp, sp)); sc[f"fullprobe__pos{pos}__HB__s{j}"] = s.astype(np.float32); sc[f"fullprobe__pos{pos}__PAP__s{j}"] = sp.astype(np.float32)
    row["full probe"] = {"hb": float(np.mean(hb)), "pap": float(np.mean(pap))}
    res["positions"][str(pos)] = row
    print(f"pos {pos:3d} {tail[p]!r:16s} PersonaJB HB {row['PersonaJB']['hb']:.3f} PAP {row['PersonaJB']['pap']:.3f} | Gaussian HB {row['Gaussian']['hb']:.3f} PAP {row['Gaussian']['pap']:.3f} | full probe HB {row['full probe']['hb']:.3f} PAP {row['full probe']['pap']:.3f}", flush=True)
for j, (tr, te) in enumerate(splits): sc[f"idx_train__s{j}"] = tr; sc[f"idx_test__s{j}"] = te
np.savez("scores_pos.npz", pair_ids_HB=np.array([r["pair_id"] for r in hv]), **sc); json.dump(res, open("detect_pos.json", "w"), indent=1)
