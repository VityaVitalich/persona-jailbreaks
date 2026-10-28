#!/usr/bin/env python3
"""
build_control_directions.py

Steering *control* directions for the "would random do this too?" question.
Every vector is unit-norm and layer 16, so steer_fixed_direction.py applies the
same alpha * RESIDUAL_NORM * v magnitude as the real arms and the controls differ
from PersonaJB/refusal/dmu only in *where* they point.

Arms (4 variants x 3 seeds):
  random_full        g ~ N(0, I_4096), unit -- a random direction anywhere in
                     activation space (~229/4096 of its norm lands in persona space).
  random_persona     uniform unit vector in the 229-dim trait subspace (SVD row
                     space of the trait matrix) -- tests whether being *in* the
                     subspace is enough, independent of which direction.
  random_traitcombo  unit(c @ V), c ~ N(0, I_229) -- a random weighting of the same
                     trait vectors PersonaJB combines (keeps trait geometry).
  shuffled_persona   unit(perm(w_trait) @ V), w_trait = coef_/scale_ of the deployed
                     detector -- the real PersonaJB weights with the trait->weight
                     pairing scrambled; same magnitude spectrum, same subspace.

offdomain_gsm8k is built separately (it needs GPU activations, see the GSM8k
collection job) and appended to this bundle afterwards.

Output matches the steering_directions_layer16.pt schema so the same runner reads it.

Usage:
  python full_trait_tools/build_control_directions.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TRAIT_MATRIX = ROOT / "full_trait_output/trait_matrix_layer16.npy"
DETECTOR = ROOT / "full_trait_output/trait_detector.pkl"
CANON = ROOT / "full_trait_output/steering_directions/steering_directions_layer16.pt"
OUT = ROOT / "full_trait_output/steering_directions/control_directions_layer16.pt"
SEEDS = [0, 1, 2]
LAYER = 16


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    # Apple's Accelerate BLAS raises spurious FP-exception flags in matmul; the
    # results (unit norms, exact in-span shares) are correct regardless.
    np.seterr(over="ignore", divide="ignore", invalid="ignore")
    V = np.load(TRAIT_MATRIX).astype(np.float64)          # (229, 4096) rows = trait dirs
    n, d = V.shape
    # Orthonormal basis of the trait subspace (row space of V) in R^d: Vt rows.
    _, _, Vt = np.linalg.svd(V, full_matrices=False)      # Vt: (229, 4096), rows orthonormal
    Q = Vt.T                                              # (4096, 229), columns span the subspace

    # Deployed detector's trait-space weights, w_trait = coef_ / scale_.
    det = pickle.load(open(DETECTOR, "rb"))
    pipe = det["pipeline"]
    clf = pipe.named_steps.get("logisticregression") or pipe.steps[-1][1]
    coef = np.asarray(clf.coef_).ravel()
    scale = getattr(pipe.named_steps.get("standardscaler", None), "scale_", None)
    w_trait = coef / scale if scale is not None else coef
    assert w_trait.shape == (n,), f"w_trait shape {w_trait.shape} != ({n},)"

    def project(vec: np.ndarray) -> np.ndarray:
        return Q @ (Q.T @ vec)

    vectors, meta = {}, {}

    def add(name: str, vec: np.ndarray, desc: str) -> None:
        u = unit(vec)
        vectors[f"{name}__full"] = torch.tensor(u, dtype=torch.float32)
        meta[f"{name}__full"] = dict(base=name, component="full", description=desc)

    for s in SEEDS:
        rng = np.random.default_rng(s)
        add(f"random_full_s{s}", rng.standard_normal(d),
            "random direction in R^4096")
        add(f"random_persona_s{s}", Q @ rng.standard_normal(n),
            "uniform random direction in the 229-dim trait subspace")
        add(f"random_traitcombo_s{s}", rng.standard_normal(n) @ V,
            "random weighting of the 229 trait vectors")
        add(f"shuffled_persona_s{s}", w_trait[rng.permutation(n)] @ V,
            "PersonaJB detector weights with trait->weight pairing permuted")

    # ── audit ────────────────────────────────────────────────────────────────
    canon = torch.load(CANON, map_location="cpu", weights_only=False)["vectors"]
    anchors = {k: v.float().numpy().astype(np.float64)
               for k, v in canon.items()
               if k in ("w_persona__full", "dmu__full", "refusal_pos1_neg__full")}
    print(f"trait subspace: {n} dims in R^{d}  (random_full expected in-span ~ {n/d:.3f})\n")
    print(f"{'arm':28s} {'norm':>6s} {'in-span':>8s} "
          f"{'cos_wp':>7s} {'cos_dmu':>8s} {'cos_ref':>8s}")
    for name, t in vectors.items():
        v = t.numpy().astype(np.float64)
        p = project(v)
        share = float(p @ p / (v @ v))
        cos = {k: float(v @ a) for k, a in anchors.items()}
        print(f"{name:28s} {np.linalg.norm(v):6.3f} {share:8.3f} "
              f"{cos.get('w_persona__full', float('nan')):7.3f} "
              f"{cos.get('dmu__full', float('nan')):8.3f} "
              f"{cos.get('refusal_pos1_neg__full', float('nan')):8.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, "meta": meta, "layer": LAYER}, OUT)
    print(f"\nwrote {len(vectors)} control vectors -> {OUT}")


if __name__ == "__main__":
    main()
