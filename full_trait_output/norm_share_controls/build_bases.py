#!/usr/bin/env python3
"""
229-direction control bases for the subspace comparison (Llama-3.1-8B, layer 16, last prompt token).
Each basis is a [229 x 4096] array of unit rows, saved to control_bases_llama.npz for
trait_pc_sweep.py --extra_bases (detectors) and subspace_tests.py (projections, Mahalanobis).

  alpaca_pca       top-229 PCs of the 797 plain Alpaca activations (centered)
  hb_pca           top-229 PCs of all HarmBench human-jailbreak activations, labels unused
                   (fit on train and test rows alike, which can only help this control)
  acts_d{i}        229 random plain Alpaca activations, unit-normalized (same form as the trait
                   vectors: unit, non-orthogonal), 5 draws
  noise_d{i}       229 shuffled-label contrasts: mean of a random half of the Alpaca activations
                   minus the mean of the other half, unit-normalized (the trait vectors are
                   diff-of-means too, here with labels that mean nothing), 5 draws
  gauss_d{i}       229 random Gaussian unit vectors, 5 draws
"""
import json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402

K, N_DRAW = 229, 5
rng = np.random.default_rng(0)
unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)


def alpaca_acts():
    rows = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
    v2 = torch.load(OUT / "roleplay_check/synthetic_v2_acts.pt", map_location="cpu")
    return np.stack([v2[r["id"]].float().numpy() for r in rows
                     if r["source"] == "alpaca" and r["frame"] == "plain"]).astype(np.float64)


def main():
    A = alpaca_acts()
    cfg = S.CFG["llama"]
    hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    Xh, _, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
    pcs = lambda X: np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:K]
    B = {"alpaca_pca": pcs(A), "hb_pca": pcs(Xh.astype(np.float64))}
    for i in range(N_DRAW):
        B[f"acts_d{i}"] = unit(A[rng.choice(len(A), K, replace=False)])
        noise = []
        for _ in range(K):
            p = rng.permutation(len(A)); h = len(A) // 2
            noise.append(A[p[:h]].mean(0) - A[p[h:]].mean(0))
        B[f"noise_d{i}"] = unit(np.stack(noise))
        B[f"gauss_d{i}"] = unit(rng.standard_normal((K, A.shape[1])))
    for n, V in B.items():
        assert V.shape == (K, A.shape[1]) and np.linalg.matrix_rank(V) == K, n
    np.savez(HERE / "control_bases_llama.npz", **{n: V.astype(np.float32) for n, V in B.items()})
    print("saved", list(B), f"(Alpaca n={len(A)}, HB n={len(Xh)})")


if __name__ == "__main__":
    main()
