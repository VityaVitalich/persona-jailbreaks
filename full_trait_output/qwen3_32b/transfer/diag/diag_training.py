"""Why PersonaJB ~ Gaussian on Qwen3-32B: C sweep x training pool (HB only vs HB+WJB) x eval (template-disjoint HB test,
random-split in-domain HB CV, transfer families). Layer 48 (and 32), last position. Saves per-prompt scores."""
import json, random, sys, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score; from sklearn.model_selection import StratifiedKFold
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"; TR = f"{Q}/transfer"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 48; POS = -1
def load(st):
    own = st not in ("hb", "wjb", "pap")
    A = torch.load(f"{TR}/acts_pos_{st}.pt" if own else f"{Q}/l48_run/acts_pos_{st}.pt")
    C = {r["pair_id"]: r for r in map(json.loads, open(f"{TR}/classified_{st}.jsonl" if own else f"{Q}/stage2_out/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    return A["X"][k, A["layers"].index(LAYER), POS].float().numpy().astype(np.float64), np.array([int(C[A["pair_ids"][i]]["jailbroken"]) for i in k]), [C[A["pair_ids"][i]] for i in k]
Xh, yh, hv = load("hb"); Xa, ya, _ = load("wjb"); FAM = ["gcg", "pair", "pap", "gptfuzz", "pez"]; E = {f: load(f)[:2] for f in FAM}
print("HB", len(yh), yh.sum(), "WJB", len(ya), ya.sum(), {f: (len(y), int(y.sum())) for f, (_, y) in E.items()}, flush=True)
names = json.load(open(f"{Q}/trait_names_layer32.json"))
T = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[LAYER].float().numpy() for n in names]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
Gs = [np.random.default_rng(i).standard_normal(T.shape) for i in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
def pool_split(seed, f=0.7):
    rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in hv}); tpl = sorted({r["jailbreak_idx"] for r in hv}); rng.shuffle(beh); rng.shuffle(tpl)
    trb, trt = set(beh[:int(len(beh) * f)]), set(tpl[:int(len(tpl) * f)]); teb, tet = set(beh[int(len(beh) * f):]), set(tpl[int(len(tpl) * f):])
    return (np.array([i for i, r in enumerate(hv) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]), np.array([i for i, r in enumerate(hv) if r["behavior_id"] in teb and r["jailbreak_idx"] in tet]))
splits = [(a, b) for a, b in (pool_split(s) for s in range(50)) if 0 < yh[b].sum() < len(b) and 0 < yh[a].sum() < len(a)][:20]
rnd = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xh, yh))
mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
res, sc = {}, {"labels_HB": yh, **{f"labels_{f}": y for f, (_, y) in E.items()}}
for nm, V in [("PersonaJB", T)] + [(f"Gaussian{i}", g) for i, g in enumerate(Gs)]:
    P = lambda X: X @ V.T; h, a = P(Xh), P(Xa); Ef = {f: P(X) for f, (X, _) in E.items()}
    for wjb in (False, True):
        for C in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            tag = f"{nm}__{'hbwjb' if wjb else 'hb'}__C{C:g}"; dis, tra, rin = [], {f: [] for f in FAM}, []
            for j, (tr, te) in enumerate(splits):
                Xt, yt = (np.vstack([h[tr], a]), np.r_[yh[tr], ya]) if wjb else (h[tr], yh[tr])
                m = mk(C).fit(Xt, yt); s = m.decision_function(h); dis.append(fold(yh[te], s[te]))
                for f in FAM: sf = m.decision_function(Ef[f]); tra[f].append(fold(E[f][1], sf)); 
                if nm in ("PersonaJB", "Gaussian0"): sc[f"{tag}__HB__s{j}"] = s.astype(np.float32); sc.update({f"{tag}__{f}__s{j}": m.decision_function(Ef[f]).astype(np.float32) for f in FAM})
            oof = np.zeros(len(yh))
            for k, (tr, te) in enumerate(rnd):
                Xt, yt = (np.vstack([h[tr], a]), np.r_[yh[tr], ya]) if wjb else (h[tr], yh[tr])
                oof[te] = mk(C).fit(Xt, yt).decision_function(h[te])
            if nm in ("PersonaJB", "Gaussian0"): sc[f"{tag}__HB_randomCV_oof"] = oof.astype(np.float32)
            row = {"hb_disjoint": float(np.mean(dis)), "hb_randomCV": float(fold(yh, oof)), **{f: float(np.mean(v)) for f, v in tra.items()}}
            row["transfer_mean"] = float(np.mean([row[f] for f in FAM])); res[tag] = row
            print(f"L{LAYER} {tag:28s} HBdisj {row['hb_disjoint']:.3f} HBrandCV {row['hb_randomCV']:.3f} " + " ".join(f"{f} {row[f]:.3f}" for f in FAM) + f" mean {row['transfer_mean']:.3f}", flush=True)
for j, (tr, te) in enumerate(splits): sc[f"idx_train__s{j}"] = tr; sc[f"idx_test__s{j}"] = te
for k, (tr, te) in enumerate(rnd): sc[f"idx_randomCV_test__f{k}"] = te
np.savez(f"scores_diag_L{LAYER}.npz", pair_ids_HB=np.array([r["pair_id"] for r in hv]), **sc); json.dump(res, open(f"diag_L{LAYER}.json", "w"), indent=1)
