#!/usr/bin/env python3
"""Independent re-implementation (no preamble / port code) of the persona-removal test on Gemma, for the audit.
Own loading, own LR pipeline, sklearn AUC; only the pool split is reused (S.pool_split, pure index logic).
Dictionaries: 240 trait vectors (PersonaJB), 240 random plain Alpaca activations (raw, and centered = minus the
mean Alpaca activation), 240 random Gaussian. Removals: trait span, random Gaussian span.
Detector: StandardScaler + LR C=10 on HB train pool + all WJB; folded AUC on HB test (splits with both classes) and
mean over the 5 Gemma attack families."""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
O = Path.home() / "persona-vector-jbing/full_trait_output"; sys.path.insert(0, str(O / "trait_pc_sweep"))
from trait_pc_sweep import pool_split  # noqa: E402
def load(rp, ap, keep=None):
    rows = [json.loads(l) for l in open(rp) if l.strip()]; A = torch.load(ap, map_location="cpu", weights_only=False)
    rows = [r for r in rows if (keep is None or keep(r)) and r.get("jailbroken") is not None and r["pair_id"] in A]
    return np.stack([A[r["pair_id"]]["30"].double().numpy() for r in rows]), np.array([int(bool(r["jailbroken"])) for r in rows]), rows
Xh, yh, hv = load(O / "harmbench_activations_gemma/classified_responses.jsonl", O / "harmbench_activations_gemma/activations.pt", lambda r: r["attack_type"] == "human_jailbreak")
Xa, ya, _ = load(O / "wildjailbreak_activations_gemma/classified_responses.jsonl", O / "wildjailbreak_activations_gemma/activations.pt")
fam = {f: load(O / f"{f.lower()}_activations_gemma_hb/classified_responses.jsonl", O / f"{f.lower()}_activations_gemma_hb/activations.pt")[:2] for f in ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]}
print("n HB", len(yh), "jb", yh.sum(), "| WJB", len(ya), ya.sum(), "|", {f: (len(y), int(y.sum())) for f, (_, y) in fam.items()})
un = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
T = un(np.load(O / "trait_pc_sweep/_gemma_ref/trait_matrix_layer30.npy").astype(np.float64)); K = len(T)
rp = [json.loads(l) for l in open(O / "roleplay_check/synthetic_v2_prompts.jsonl")]
G = torch.load(Path(__file__).resolve().parent.parent / "gemma_acts.pt", map_location="cpu")
Al = np.stack([G["rp:" + r["id"]].double().numpy() for r in rp if r["source"] == "alpaca" and r["frame"] == "plain"])
rng = np.random.default_rng(123); pick = rng.choice(len(Al), K, replace=False)
D = {"PersonaJB traits": T, "Alpaca acts (raw)": un(Al[pick]), "Alpaca acts (centered)": un(Al[pick] - Al.mean(0)),
     "Random Gaussian": un(rng.standard_normal((K, T.shape[1])))}
Qp = np.linalg.qr(T.T)[0]; Qr = np.linalg.qr(rng.standard_normal((K, T.shape[1])).T)[0]
splits = [tuple(map(np.array, pool_split(hv, s))) for s in range(50)]
splits = [(tr, te) for tr, te in splits if 0 < yh[te].sum() < len(te)]
fa = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
def det(V):
    hb, xf = [], []
    for tr, te in splits:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=4000)).fit(np.vstack([Xh[tr], Xa]) @ V.T, np.r_[yh[tr], ya])
        hb.append(fa(yh[te], m.decision_function(Xh[te] @ V.T))); xf.append(np.mean([fa(y, m.decision_function(X @ V.T)) for X, y in fam.values()]))
    return np.mean(hb), np.mean(xf)
print(f"{len(splits)} splits\n{'dictionary':24s} {'share in trait span':>19s} | {'full HB xfer':>13s} | {'persona removed':>15s} | {'random span removed':>19s}")
for n, V in D.items():
    sh = float(np.mean(np.sum((V @ Qp) ** 2, 1))); f = det(V)
    p = det(un(V - V @ Qp @ Qp.T)) if n != "PersonaJB traits" else (np.nan, np.nan); r = det(un(V - V @ Qr @ Qr.T))
    print(f"{n:24s} {sh:19.3f} | {f[0]:.3f} {f[1]:.3f} | {p[0]:.3f} {p[1]:.3f}     | {r[0]:.3f} {r[1]:.3f}", flush=True)
