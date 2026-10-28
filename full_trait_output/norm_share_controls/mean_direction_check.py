#!/usr/bin/env python3
"""Why do single random activations read out compliance? Check the shared mean component:
cos of each dictionary vector with its dictionary mean, AUC of h . mean, AUC of ||h||, and the AUC of
single vectors after removing the mean direction. Saved activations only."""
import sys, warnings
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
D = {"Persona traits": T, "Wikipedia": np.concatenate([eb[f"wiki_d{i}"] for i in range(5)]),
     "No Robots": np.concatenate([eb[f"nr_d{i}"] for i in range(5)]),
     "Alpaca": np.concatenate([cb[f"acts_d{i}"] for i in range(5)])}
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
fold = lambda a: np.maximum(a, 1 - a)
unit = lambda v: v / np.linalg.norm(v)
print(f"HarmBench prompts: AUC of ||h|| = {fold(rank_auc(yh, np.linalg.norm(Xh, axis=1)[:, None]))[0]:.3f}")
mh = unit(Xh.mean(0))
print(f"HarmBench mean activation direction: AUC of h.mean = {fold(rank_auc(yh, (Xh @ mh)[:, None]))[0]:.3f}")
print(f"\n{'dictionary':16s} median cos(v, dict mean)  AUC(h.mean)  median single-v AUC  -> after removing mean dir")
for n, V in D.items():
    m = unit(V.mean(0)); c = V @ m
    a_mean = fold(rank_auc(yh, (Xh @ m)[:, None]))[0]
    a = fold(rank_auc(yh, Xh @ V.T))
    R = V - np.outer(V @ m, m); R /= np.linalg.norm(R, axis=1, keepdims=True)
    a_r = fold(rank_auc(yh, Xh @ R.T))
    print(f"{n:16s} {np.median(c):24.3f} {a_mean:12.3f} {np.median(a):20.3f} {np.median(a_r):16.3f}")

sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
from llama_50seed_span import load_direction  # noqa: E402
ref = unit(load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt").astype(np.float32))
a_ref = rank_auc(yh, (Xh @ ref)[:, None])[0]
print(f"\nrefusal direction alone: AUC(h.ref) = {fold(a_ref):.3f} (signed {a_ref:.3f}, jailbroken = higher)")
for n, V in D.items():
    m = unit(V.mean(0)); a = rank_auc(yh, (Xh @ m)[:, None])[0]
    print(f"{n:16s} mean dir: signed AUC {a:.3f} (>0.5 = jailbroken prompts project higher)   cos(mean, refusal) {m @ ref:+.3f}")
