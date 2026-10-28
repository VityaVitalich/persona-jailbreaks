import json, random, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = torch.load("diag_acts.pt"); C = {r["pair_id"]: r for r in map(json.loads, open("../stage2_out/classified_hb.jsonl"))}
idx = [i for i, p in enumerate(D["pair_ids"]) if C[p]["attack_type"] == "human_jailbreak"]; hv = [C[D["pair_ids"][i]] for i in idx]
y = np.array([int(r["jailbroken"]) for r in hv]); X = D["X"][idx].float().numpy()
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
def split(seed, f=.7):
    rng = random.Random(seed); b = sorted({r["behavior_id"] for r in hv}); t = sorted({r["jailbreak_idx"] for r in hv}); rng.shuffle(b); rng.shuffle(t)
    trb, trt = set(b[:int(len(b)*f)]), set(t[:int(len(t)*f)])
    return (np.array([i for i, r in enumerate(hv) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]), np.array([i for i, r in enumerate(hv) if r["behavior_id"] not in trb and r["jailbreak_idx"] not in trt]))
S = [s for s in (split(k) for k in range(10)) if 0 < y[s[1]].sum() < len(s[1])]
old = torch.load("../stage2_out/acts_hb.pt"); o = np.stack([old[hv[i]["pair_id"]]["32"].float().numpy() for i in range(len(hv))])
j = D["slots"].index("think_L32_pos-1"); print("stored vs diag L32 pos-1 min cos", (np.sum(o*X[:, j], 1)/np.linalg.norm(o, axis=1)/np.linalg.norm(X[:, j], axis=1)).min().round(5))
for k, s in enumerate(D["slots"]):
    a = [fold(y[te], make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(X[tr, k], y[tr]).decision_function(X[te, k])) for tr, te in S]
    print(f"{s:20s} full-act LR test AUC {np.mean(a):.3f}  (min {np.min(a):.2f})", flush=True)
