#!/usr/bin/env python3
"""
plot_per_trait_readout.py

Paper figure: per-token SINGLE-TRAIT projection around the pre-generation
readout token (default: egalitarian, the most predictive trait). Unlike the
aggregate readout figure (all-229-trait classifier logit), the y-axis projects
the layer-16 activation onto one trait direction, so no classifier is
involved — group separation is read directly off the projection.

Two groups, same behavioral definitions as the aggregate figure:
  * jailbreak that failed (refused)     — per_position_probe_recast_data.npz
  * jailbreak that succeeded (complied) — per_position_probe_recast_data.npz
Both cover all 20 human-jailbreak templates; labels come from the response
judge. Bands are 95% CIs of the per-position mean. (Per-token trait
projections for direct requests were never collected, only their classifier
logits — so the direct group of the aggregate figure has no per-trait analog.)

Trait column ordering comes from all_traits_sweep_v2/best_model.pkl (same
trait_matrix as the detector used at collection time).

Usage:
  python3 full_trait_tools/plot_per_trait_readout.py \
      [--trait egalitarian] \
      [--output_path full_trait_output/plots/per_trait_readout.pdf]
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# journal palette; same jailbroken/refused semantics as the PCA panels
C_JAILBRK = "#D28181"
C_REFUSED = "#6A90BA"
GRID    = "#E7E7E7"
SPINE   = "#C9C9C9"
TEXT_M  = "#555555"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
        default="full_trait_output/per_position_probe_recast_data.npz")
    parser.add_argument("--model_pkl", type=str,
        default="full_trait_output/all_traits_sweep_v2/best_model.pkl")
    parser.add_argument("--trait", type=str, default="egalitarian")
    parser.add_argument("--min_n", type=int, default=10)
    parser.add_argument("--output_path", type=str,
        default="full_trait_output/plots/per_trait_readout.pdf")
    args = parser.parse_args()

    z = np.load(args.data)
    X, Y, W = z["X"], z["Y"], int(z["W"])            # X: [n, W+1, 229]
    names = pickle.load(open(args.model_pkl, "rb"))["trait_names"]
    ti = names.index(args.trait)
    x = np.arange(-W, 1)

    groups = [
        ("jailbreak that failed (refused)",     C_REFUSED, X[Y == 0][:, :, ti]),
        ("jailbreak that succeeded (complied)", C_JAILBRK, X[Y == 1][:, :, ti]),
    ]

    fig, ax = plt.subplots(figsize=(4.7, 3.25), dpi=200)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_M, labelsize=10, length=0)

    ax.axvspan(-4.5, 0.5, color="0.93", zorder=0)

    for lab, color, P in groups:
        n = np.sum(~np.isnan(P), axis=0)
        m = np.nanmean(P, axis=0)
        se = np.nanstd(P, axis=0) / np.sqrt(np.maximum(n, 1))
        keep = n >= args.min_n
        ax.plot(x[keep], m[keep], color=color, lw=2.0, label=lab, zorder=3)
        ax.fill_between(x[keep], (m - 1.96 * se)[keep], (m + 1.96 * se)[keep],
                        color=color, alpha=0.20, linewidth=0, zorder=2)
        ax.plot([0], [m[-1]], marker="*", ms=13, color=color,
                markeredgecolor="white", markeredgewidth=0.7, zorder=5)

    t = ax.text(-2.0, 0.97, "assistant header\n+ readout", ha="center",
                va="top", fontsize=8.5, color=TEXT_M,
                transform=ax.get_xaxis_transform())
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlabel("token position relative to the readout token",
                  fontsize=11, color=TEXT_M)
    ax.set_ylabel(f"{args.trait} projection (layer 16)",
                  fontsize=11, color=TEXT_M)

    leg = ax.legend(fontsize=9, loc="lower left", framealpha=0.9,
                    edgecolor=SPINE, borderpad=0.5, handlelength=1.8)
    leg.get_frame().set_facecolor("white")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
