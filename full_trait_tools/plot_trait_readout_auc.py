#!/usr/bin/env python3
"""
plot_trait_readout_auc.py

Paper figure: per-token-position separation between succeeded and failed
jailbreaks for several single-trait projections, measured as worked-vs-failed
AUC (scale-invariant, so traits share one axis; the paper's headline metric).

Data: per_position_probe_recast_data.npz (X: [n, W+1, 229] layer-16 trait
projections over the last W+1 prompt tokens, all 20 human-jailbreak
templates; Y = jailbroken label from the response judge). Each trait's
orientation is fixed ONCE from its readout-token AUC and applied to every
position — no per-position flipping, so content positions can honestly sit
at chance. Bands: analytic 95% CI (Hanley & McNeil 1982).

Usage:
  python3 full_trait_tools/plot_trait_readout_auc.py \
      [--personas egalitarian universalist progressive principled] \
      [--output_path full_trait_output/plots/trait_readout_auc.pdf]
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from sklearn.metrics import roc_auc_score

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


def auc_ci(y, s):
    """AUC with Hanley-McNeil 95% CI half-width."""
    a = roc_auc_score(y, s)
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                  + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return a, 1.96 * se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
        default="full_trait_output/per_position_probe_recast_data.npz")
    parser.add_argument("--model_pkl", type=str,
        default="full_trait_output/all_traits_sweep_v2/best_model.pkl")
    parser.add_argument("--personas", nargs="+", default=PERSONAS)
    parser.add_argument("--output_path", type=str,
        default="full_trait_output/plots/trait_readout_auc.pdf")
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

    ax.axvspan(-4.5, 0.5, color="0.93", zorder=0)
    ax.axhline(0.5, ls=(0, (6, 4)), color="#8A8A8A", lw=1.2, zorder=1)

    for trait, (color, marker) in zip(args.personas, SERIES):
        P = X[:, :, names.index(trait)]              # [n, W+1]
        # one global orientation per trait, fixed at the readout token
        ok = ~np.isnan(P[:, -1])
        if roc_auc_score(Y[ok], P[ok, -1]) < 0.5:
            P = -P
        aucs, cis = [], []
        for p in range(W + 1):
            ok = ~np.isnan(P[:, p])
            a, ci = auc_ci(Y[ok], P[ok, p])
            aucs.append(a)
            cis.append(ci)
        aucs, cis = np.array(aucs), np.array(cis)
        ax.plot(x, aucs, color=color, lw=1.8, marker=marker, ms=3.4,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.4, label=trait, zorder=3)
        ax.fill_between(x, aucs - cis, aucs + cis, color=color, alpha=0.18,
                        linewidth=0, zorder=2)

    t = ax.text(0.985, 0.504, "chance", color=TEXT_M, fontsize=9.5,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    t = ax.text(-2.0, 0.97, "assistant header\n+ readout", ha="center",
                va="top", fontsize=8.5, color=TEXT_M,
                transform=ax.get_xaxis_transform())
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlabel("token position relative to the readout token",
                  fontsize=11, color=TEXT_M)
    ax.set_ylabel("succeeded-vs-failed AUC", fontsize=11, color=TEXT_M)

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
