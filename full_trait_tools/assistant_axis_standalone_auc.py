#!/usr/bin/env python3
"""
assistant_axis_standalone_auc.py

Standalone jailbreak-detection AUC of the assistant axis (PC1 of the persona/trait
space), using the same held-out-seed protocol as pca_direction_decomposition.py so the
number is directly comparable to the per-PC standalone AUCs. The assistant axis is a
fixed direction (no fitting); for each seed we project the test-split layer-L
activations onto it and take safe_auc (orientation-invariant, like the PCs).

Reports mean +/- std over seeds per layer, plus the full-set AUC for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_all_traits_sweep_v2 import (
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    safe_auc,
    split_by_pool,
)

TRAIN_FRAC = 0.7
N_SEEDS = 101


def load_axis(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    v = data["axis"][layer].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",
                        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",
                        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--axis_path",
                        default="full_trait_output/paper_data/analysis/jailbreak_pca_inputs/assistant_axis_pc1.pt")
    parser.add_argument("--output_dir", default="full_trait_output/assistant_axis_standalone_auc")
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 28])
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===", flush=True)
    human_rows = [
        r for r in load_jsonl(Path(args.human_classified_path))
        if r.get("attack_type") == "human_jailbreak"
    ]
    human_acts = torch.load(Path(args.human_activations_path), map_location="cpu", weights_only=False)

    results = {}
    for layer in args.layers:
        x_raw_h, y_h, human_valid = build_activation_matrix(human_rows, human_acts, layer)
        axis = load_axis(Path(args.axis_path), layer)
        proj = x_raw_h @ axis  # [n]

        full_auc = safe_auc(y_h, proj)

        seed_aucs = []
        n_used = 0
        for seed in range(args.n_seeds):
            train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, args.train_frac, seed)
            train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
            if not test_idx or len(np.unique(y_h[test_idx])) < 2:
                continue
            n_used += 1
            seed_aucs.append(safe_auc(y_h[test_idx], proj[test_idx]))

        res = {
            "layer": layer,
            "n_pairs": int(len(y_h)),
            "jb_rate": float(y_h.mean()),
            "n_seeds_used": n_used,
            "standalone_auc_mean": float(np.nanmean(seed_aucs)),
            "standalone_auc_std": float(np.nanstd(seed_aucs)),
            "full_set_auc": float(full_auc),
        }
        results[f"layer_{layer}"] = res
        print(f"  layer {layer}: standalone AUC = {res['standalone_auc_mean']:.4f} "
              f"+/- {res['standalone_auc_std']:.4f} over {n_used} seeds "
              f"(full-set {full_auc:.4f}, n={res['n_pairs']}, jb={res['jb_rate']:.3f})",
              flush=True)

    out = {
        "method": "assistant_axis_standalone_auc",
        "protocol": "held-out-seed standalone AUC, matching pca_direction_decomposition",
        "n_seeds": args.n_seeds,
        "train_frac": args.train_frac,
        "axis_path": args.axis_path,
        "results": results,
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
