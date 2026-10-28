#!/usr/bin/env python3
"""
Single-vector readout: score = h . v for ONE unit vector v (no classifier, no fitting, no subspace),
folded AUC max(a, 1-a) on all HarmBench human_jailbreak rows and on each transfer family.
Distribution over every vector of each dictionary. Saved activations only.
Writes single_vector_auc.json (every vector's AUCs) and prints a summary.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(HERE))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402

cfg = S.CFG["llama"]
T = np.load(cfg["T"]).astype(np.float32); T /= np.linalg.norm(T, axis=1, keepdims=True)
cb, eb = np.load(HERE / "control_bases_llama.npz"), np.load(HERE / "extra_bases_llama.npz")
D = {"Persona traits": T,
     "Wikipedia activations": np.concatenate([eb[f"wiki_d{i}"] for i in range(5)]),
     "No Robots activations": np.concatenate([eb[f"nr_d{i}"] for i in range(5)]),
     "Alpaca activations": np.concatenate([cb[f"acts_d{i}"] for i in range(5)]),
     "Random Gaussian": np.concatenate([cb[f"gauss_d{i}"] for i in range(5)])}
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
fam = {}
for f, (rp, ap) in cfg["fams"].items():
    X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)
    fam[f] = (X, y)
fold = lambda a: np.maximum(a, 1 - a)
res = {}
print(f"{'dictionary':24s} {'n':>5s} | HarmBench folded AUC: median  p90   max | transfer (mean of 5 fams): median  p90   max")
for n, V in D.items():
    V = V.astype(np.float32)
    h = fold(rank_auc(yh, Xh @ V.T))
    xf = np.mean([fold(rank_auc(y, X @ V.T)) for X, y in fam.values()], axis=0)
    res[n] = {"hb": h.tolist(), "transfer": xf.tolist()}
    q = lambda a: f"{np.median(a):.3f} {np.quantile(a, .9):.3f} {a.max():.3f}"
    print(f"{n:24s} {len(V):5d} |                     {q(h)} |                              {q(xf)}")
json.dump(res, open(HERE / "single_vector_auc.json", "w"))
