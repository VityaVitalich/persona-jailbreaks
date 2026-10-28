#!/usr/bin/env python3
"""Is 'jailbroken' vs 'benign-complied' linearly separable at all in the trait space (229 trait
projections) or in raw L16 activations? Logistic-regression probes on the synthetic set, 5-fold CV
grouped by JailbreakBench behaviour (harmful and benign versions of a behaviour stay in the same fold),
C chosen by inner grouped CV. Also the PersonaJB score itself as reference. Writes part3_probe.json."""
import json
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
meta = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "synthetic_prompts.jsonl")}
jud = {json.loads(l)["id"]: json.loads(l)["complied"] for l in open(HERE / "synthetic_judged.jsonl")}
acts = torch.load(HERE / "synthetic_acts.pt", map_location="cpu")
T = np.load(HERE.parent / "trait_matrix_layer16.npy").astype(np.float32); T /= np.linalg.norm(T, axis=1, keepdims=True)
d = np.load(HERE / "part2_scores.npz", allow_pickle=True); pj = dict(zip(d["ids"], d["scores"].mean(0)))
out = {}
for a_name, a_sel, b_name, b_sel in [("jailbroken", lambda r, c: r["harm"] == "harmful" and c, "benign_complied", lambda r, c: r["harm"] == "benign" and c),
                                     ("jailbroken", lambda r, c: r["harm"] == "harmful" and c, "refused", lambda r, c: r["harm"] == "harmful" and not c)]:
    ids = [i for i, r in meta.items() if a_sel(r, jud[i]) or b_sel(r, jud[i])]
    y = np.array([int(a_sel(meta[i], jud[i])) for i in ids]); g = np.array([meta[i]["jbb_index"] for i in ids])
    X = np.stack([acts[i].float().numpy() for i in ids]).astype(np.float64); feats = {"raw L16 activation": X, "trait projections (229)": X @ T.T}
    res = {"n": [int(y.sum()), int((1 - y).sum())], "PersonaJB score (fixed)": float(max(roc_auc_score(y, [pj[i] for i in ids]), 1 - roc_auc_score(y, [pj[i] for i in ids])))}
    for fname, F in feats.items():
        s = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(F, y, g):
            inner = list(GroupKFold(4).split(F[tr], y[tr], g[tr]))  # behaviour-grouped inner CV for C
            m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], cv=inner, max_iter=3000, scoring="roc_auc"))
            m.fit(F[tr], y[tr]); s[te] = m.predict_proba(F[te])[:, 1]
        res[fname] = float(roc_auc_score(y, s))
    out[f"{a_name} vs {b_name}"] = res; print(f"{a_name} vs {b_name}:", json.dumps(res), flush=True)
json.dump(out, open(HERE / "part3_probe.json", "w"), indent=1)
