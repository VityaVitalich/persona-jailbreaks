#!/usr/bin/env python3
"""
plot_trait_readout_distance.py

Paper figure: per-token-position DISTANCE between the succeeded and failed
jailbreak groups in several single-trait projections. Distance = standardized
mean difference (Cohen's d): (mean_succeeded - mean_failed) / pooled SD at
each position, so traits with different projection scales share one axis
while the y-value stays a literal gap between the two group means.

Data: per_position_probe_recast_data.npz (X: [n, W+1, 229] layer-16 trait
projections over the last W+1 prompt tokens, all 20 human-jailbreak
templates; Y = jailbroken label from the response judge). Each trait's sign
is fixed ONCE from its readout-token gap and applied to every position — no
per-position flipping, so content positions can honestly sit at (or cross)
zero. Bands: analytic 95% CI of d.

Usage:
  python3 full_trait_tools/plot_trait_readout_distance.py \
      [--personas egalitarian universalist progressive principled] \
      [--output_path full_trait_output/plots/trait_readout_distance.pdf]
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

PERSONAS = ["egalitarian", "universalist", "progressive", "principled"]

# journal palette (matches plot_persona_layer_dynamics.py);
# markers double as secondary encoding
SERIES = [
    ("#1D5B8C", "o"),
    ("#A6403A", "s"),
    ("#42926A", "^"),
    ("#C88A2D", "D"),
]
GRID   = "#E7E7E7"
SPINE  = "#C9C9C9"
TEXT_M = "#555555"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def cohens_d(a, b):
    """d = (mean(a) - mean(b)) / pooled SD, with 95% CI half-width."""
    n1, n0 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n0 - 1) * np.var(b, ddof=1))
                 / (n1 + n0 - 2))
    d = (np.mean(a) - np.mean(b)) / sp
    se = np.sqrt((n1 + n0) / (n1 * n0) + d * d / (2 * (n1 + n0)))
    return d, 1.96 * se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
        default="full_trait_output/per_position_probe_recast_data.npz")
    parser.add_argument("--model_pkl", type=str,
        default="full_trait_output/all_traits_sweep_v2/best_model.pkl")
    parser.add_argument("--personas", nargs="+", default=PERSONAS)
    parser.add_argument("--output_path", type=str,
        default="full_trait_output/plots/trait_readout_distance.pdf")
    args = parser.parse_args()

    z = np.load(args.data)
    X, Y, W = z["X"], z["Y"], int(z["W"])            # X: [n, W+1, 229]
    names = pickle.load(open(args.model_pkl, "rb"))["trait_names"]
    x = np.arange(-W, 1)

    fig, ax = plt.subplots(figsize=(4.7, 3.25), dpi=200)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_M, labelsize=10, length=0)
    ax.tick_params(axis="x", length=3, color=SPINE)

    ax.axvspan(-4.5, 0.5, color="0.93", zorder=0)
    ax.axhline(0.0, ls=(0, (6, 4)), color="#8A8A8A", lw=1.2, zorder=1)

    for trait, (color, marker) in zip(args.personas, SERIES):
        P = X[:, :, names.index(trait)]              # [n, W+1]
        ds, cis = [], []
        for p in range(W + 1):
            ok = ~np.isnan(P[:, p])
            d, ci = cohens_d(P[ok & (Y == 1), p], P[ok & (Y == 0), p])
            ds.append(d)
            cis.append(ci)
        ds, cis = np.array(ds), np.array(cis)
        # one global orientation per trait, fixed at the readout token
        if ds[-1] < 0:
            ds = -ds
        ax.plot(x, ds, color=color, lw=1.8, marker=marker, ms=3.4,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.4, label=trait, zorder=3)
        ax.fill_between(x, ds - cis, ds + cis, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot([0], [ds[-1]], marker="*", ms=11, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    ax.set_ylim(-1.35, 2.45)

    t = ax.text(-2.0, 2.33, "assistant\ntemplate", ha="center", va="top",
                fontsize=8.5, color=TEXT_M)
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # the shaded positions are the chat template itself, identical for every
    # prompt — write the tokens (special-token delimiters stripped) vertically
    # in the free space below the curves, so the bottom axis keeps the compact
    # numeric ticks of the paired panel
    template_tokens = ["eot_id", "start_header_id", "assistant",
                       "end_header_id", r"\n\n"]
    for pos, tok in zip(range(-4, 1), template_tokens):
        t = ax.text(pos, -1.28, tok, rotation=90, ha="center", va="bottom",
                    fontsize=6.0, color=TEXT_M, fontfamily="monospace",
                    zorder=2.5)
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_xticks([-25, -20, -15, -10, -5, 0])

    ax.set_xlabel("token position relative to the readout token",
                  fontsize=11, color=TEXT_M)
    ax.set_ylabel("succeeded−failed normalized distance",
                  fontsize=10, color=TEXT_M)

    leg = ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9,
                    edgecolor=SPINE, borderpad=0.5, handlelength=2.0)
    leg.get_frame().set_facecolor("white")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
