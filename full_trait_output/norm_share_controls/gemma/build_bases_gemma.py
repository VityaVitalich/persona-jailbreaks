#!/usr/bin/env python3
"""Gemma-4-31B-it counterpart of build_bases.py + the base part of build_steer_wiki.py: K = 240 (the number of Gemma
trait vectors) unit-row dictionaries of layer-30 last-token activations (gemma_acts.pt, gemma_acts.py), same seeds
and draws as the Llama files.
  control_bases_gemma.npz  alpaca_pca, hb_pca, acts_d0..4, noise_d0..4, gauss_d0..4   (rng seed 0, as build_bases.py)
  extra_bases_gemma.npz    wiki_d0..4 (raw text), nr_d0..4 (No Robots, chat)          (rng seed 1, as build_steer_wiki.py)
Also checks the readout: cosine of the 24 re-embedded HarmBench prompts (verify:*) with the stored activations."""
import json, sys
from pathlib import Path
import numpy as np, torch
G = Path(__file__).resolve().parent; OUT = G.parent.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402
unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
K = np.load(S.CFG["gemma"]["T"]).shape[0]; N_DRAW = 5
acts = torch.load(G / "gemma_acts.pt", map_location="cpu")
stored = torch.load(S.CFG["gemma"]["human"][1], map_location="cpu", weights_only=False)
cos = [float(torch.nn.functional.cosine_similarity(v.float(), stored[int(k.split(":")[1])]["30"].float(), dim=0)) for k, v in acts.items() if k.startswith("verify:")]
print(f"readout check: {len(cos)} HarmBench prompts, cosine with stored min {min(cos):.4f} median {np.median(cos):.4f}")
assert min(cos) > 0.99, cos
rows = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
A = np.stack([acts["rp:" + r["id"]].float().numpy() for r in rows if r["source"] == "alpaca" and r["frame"] == "plain"]).astype(np.float64)
hb = [r for r in S.load_jsonl(S.CFG["gemma"]["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, _, _ = S.build_matrix(hb, stored, 30)
pcs = lambda X: np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:K]
rng = np.random.default_rng(0)
B = {"alpaca_pca": pcs(A), "hb_pca": pcs(Xh.astype(np.float64))}
for i in range(N_DRAW):
    B[f"acts_d{i}"] = unit(A[rng.choice(len(A), K, replace=False)])
    noise = []
    for _ in range(K):
        p = rng.permutation(len(A)); h = len(A) // 2
        noise.append(A[p[:h]].mean(0) - A[p[h:]].mean(0))
    B[f"noise_d{i}"] = unit(np.stack(noise))
    B[f"gauss_d{i}"] = unit(rng.standard_normal((K, A.shape[1])))
E = {}
for src, pre in (("wiki", "wiki_"), ("nr", "nr_")):
    ids = sorted((k for k in acts if k.startswith(pre)), key=lambda s: int(s.split("_")[1]))
    M = np.stack([acts[i].float().numpy() for i in ids]).astype(np.float64)
    r1 = np.random.default_rng(1)
    for d in range(N_DRAW): E[f"{src}_d{d}"] = unit(M[r1.choice(len(M), K, replace=False)])
for n, V in {**B, **E}.items():
    assert V.shape == (K, A.shape[1]) and np.linalg.matrix_rank(V) == K, n
np.savez(G / "control_bases_gemma.npz", **{n: V.astype(np.float32) for n, V in B.items()})
np.savez(G / "extra_bases_gemma.npz", **{n: V.astype(np.float32) for n, V in E.items()})
print("saved", list(B), list(E), f"(K={K}, Alpaca n={len(A)}, HB n={len(Xh)})")
