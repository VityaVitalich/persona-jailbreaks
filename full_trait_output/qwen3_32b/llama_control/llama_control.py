"""Control: the Qwen detector protocol (transfer/diag/diag_training.py) on Llama-3.1-8B's shared layer-16 activations and
judge labels (public/persona_jailbreaks). If it reproduces the paper's Llama transfer (~.76 trait / ~.71 raw), the code is
fine. Also: how concentrated HB human-jailbreak positives are over templates, for Llama, Gemma (from the copied labels
when present) and Qwen. Saves per-prompt scores to scores_llama_control.npz, table to llama_control.json."""
import json, random, collections, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score; from sklearn.model_selection import StratifiedKFold
D = "data"; FAM = ["gcg", "pair", "pap", "gptfuzz", "pez"]; LAYER = "16"
def load(st):
    f = f"{D}/{st}/classified_responses.jsonl"
    try: R = [json.loads(l) for l in open(f)]
    except FileNotFoundError: R = [json.loads(l) for l in open(f"{D}/{st}/responses.jsonl")]
    A = torch.load(f"{D}/{st}/activations.pt", map_location="cpu", weights_only=False)
    R = [r for r in R if r.get("jailbroken") is not None and r["pair_id"] in A and (st != "harmbench" or r["attack_type"] == "human_jailbreak")]
    return np.stack([A[r["pair_id"]][LAYER].float().numpy() for r in R]).astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R
Xh, yh, hv = load("harmbench"); Xa, ya, _ = load("wildjailbreak"); E = {f: load(f)[:2] for f in FAM}
print("HB", len(yh), yh.sum(), "WJB", len(ya), ya.sum(), {f: (len(y), int(y.sum())) for f, (_, y) in E.items()}, flush=True)
T = np.load(f"{D}/trait_matrix_layer16.npy").astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True); print("trait matrix", T.shape)
Gs = [np.random.default_rng(i).standard_normal(T.shape) for i in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
def pool_split(seed, f=0.7):
    rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in hv}); tpl = sorted({r["jailbreak_idx"] for r in hv}); rng.shuffle(beh); rng.shuffle(tpl)
    trb, trt = set(beh[:int(len(beh) * f)]), set(tpl[:int(len(tpl) * f)]); teb, tet = set(beh[int(len(beh) * f):]), set(tpl[int(len(tpl) * f):])
    return (np.array([i for i, r in enumerate(hv) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]), np.array([i for i, r in enumerate(hv) if r["behavior_id"] in teb and r["jailbreak_idx"] in tet]))
splits = [(a, b) for a, b in (pool_split(s) for s in range(50)) if 0 < yh[b].sum() < len(b) and 0 < yh[a].sum() < len(a)][:20]
rnd = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xh, yh))
mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
res, sc = {}, {"labels_HB": yh, "pair_ids_HB": np.array([r["pair_id"] for r in hv]), **{f"labels_{f}": y for f, (_, y) in E.items()}}
for nm, V, C in [("PersonaJB", T, 10.0)] + [(f"Gaussian{i}", g, 10.0) for i, g in enumerate(Gs)] + [("fullprobe_C0.01", None, 0.01), ("fullprobe_C1", None, 1.0)]:
    P = (lambda X: X @ V.T) if V is not None else (lambda X: X); h, a = P(Xh), P(Xa); Ef = {f: P(X) for f, (X, _) in E.items()}
    for wjb in (False, True):
        tag = f"{nm}__{'hbwjb' if wjb else 'hb'}"; dis, tra = [], {f: [] for f in FAM}
        fit = lambda idx: mk(C).fit(np.vstack([h[idx], a]) if wjb else h[idx], np.r_[yh[idx], ya] if wjb else yh[idx])
        for j, (tr, te) in enumerate(splits):
            m = fit(tr); s = m.decision_function(h); dis.append(fold(yh[te], s[te])); sc[f"{tag}__HB__s{j}"] = s.astype(np.float32)
            for f in FAM: sf = m.decision_function(Ef[f]); tra[f].append(fold(E[f][1], sf)); sc[f"{tag}__{f}__s{j}"] = sf.astype(np.float32)
        oof = np.zeros(len(yh))
        for tr, te in rnd: oof[te] = fit(tr).decision_function(h[te])
        sc[f"{tag}__HB_randomCV_oof"] = oof.astype(np.float32)
        row = {"hb_randomCV": float(fold(yh, oof)), "hb_disjoint": float(np.mean(dis)), **{f: float(np.mean(v)) for f, v in tra.items()}}; row["transfer_mean"] = float(np.mean([row[f] for f in FAM])); res[tag] = row
        print(f"{tag:24s} HBcv {row['hb_randomCV']:.3f} HBdisj {row['hb_disjoint']:.3f} " + " ".join(f"{f} {row[f]:.3f}" for f in FAM) + f" mean {row['transfer_mean']:.3f}", flush=True)
for j, (tr, te) in enumerate(splits): sc[f"idx_train__s{j}"] = tr; sc[f"idx_test__s{j}"] = te
np.savez("scores_llama_control.npz", **sc); json.dump(res, open("llama_control.json", "w"), indent=1)
