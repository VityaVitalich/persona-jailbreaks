#!/usr/bin/env python3
"""Independent check of the persona-removed refusal / dmu detection numbers: plain sklearn roc_auc_score, per family,
signed (oriented so that the intact direction is > 0.5 on HarmBench) and folded. Trait matrix from trait_detector.pkl
(the deployed detector) and from trait_matrix_layer16.npy, to rule out a cache mismatch."""
import json, pickle, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
from llama_50seed_span import load_direction  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
cfg = S.CFG["llama"]
T1 = np.load(cfg["T"]).astype(np.float64)
T2 = np.asarray(pickle.load(open(OUT / "trait_detector.pkl", "rb"))["trait_matrix"], np.float64)
print("trait matrices identical up to row scale:", np.allclose(T1 / np.linalg.norm(T1, axis=1, keepdims=True), T2 / np.linalg.norm(T2, axis=1, keepdims=True), atol=1e-5))
Q = np.linalg.qr((T2 / np.linalg.norm(T2, axis=1, keepdims=True)).T)[0]
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16); Xh = Xh.astype(np.float64)
fam = {}
for f, (rp, ap) in cfg["fams"].items():
    X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16); fam[f] = (X.astype(np.float64), y)
ref = unit(-load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt").astype(np.float64))
dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))
for name, v in (("refusal (-Arditi)", ref), ("dmu (all HB rows)", dmu)):
    w = unit(v - Q @ (Q.T @ v))
    print(f"\n{name}: share in persona span {np.sum((Q.T @ v) ** 2):.3f}; check |Q^T w| = {np.linalg.norm(Q.T @ w):.2e}; cos(v, w) {v @ w:.3f}")
    print(f"  {'set':10s} {'n (jb)':>11s} | {'intact signed':>13s} {'removed signed':>14s} | {'intact folded':>13s} {'removed folded':>14s}")
    rows = [("HarmBench", Xh, yh)] + [(f, X, y) for f, (X, y) in fam.items()]
    fi, fr = [], []
    for f, X, y in rows:
        a, b = roc_auc_score(y, X @ v), roc_auc_score(y, X @ w)
        print(f"  {f:10s} {len(y):5d} ({y.sum():4d}) | {a:13.3f} {b:14.3f} | {max(a, 1 - a):13.3f} {max(b, 1 - b):14.3f}")
        if f != "HarmBench": fi.append(max(a, 1 - a)); fr.append(max(b, 1 - b))
    print(f"  mean transfer folded: intact {np.mean(fi):.3f} -> removed {np.mean(fr):.3f}")
