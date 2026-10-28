#!/usr/bin/env python3
"""
plot_persona_layer_dynamics.py

Paper figure: per-layer single-trait detection AUC for the top predictive
personas, plotted from the results.json emitted by persona_layer_dynamics.py
(so restyling never re-runs the projection/CV compute).

Shows only the individual persona curves plus a chance line and a marker for
the paper's default layer 16; the all-229 logreg and mean-single-trait
baselines in results.json are not drawn.

Usage:
  python3 full_trait_tools/plot_persona_layer_dynamics.py \
      [--results full_trait_output/persona_layer_dynamics/results.json] \
      [--personas egalitarian universalist progressive principled] \
      [--output_path full_trait_output/plots/persona_layer_dynamics.pdf]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PERSONAS = ["egalitarian", "universalist", "progressive", "principled"]
DEFAULT_LAYER = 16

# journal palette (matches plot_activation_scatter.py / plot_trait_auc_dotplot.py);
# all pairs CVD-checked, markers double as secondary encoding
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
        default="full_trait_output/persona_layer_dynamics/results.json")
    parser.add_argument("--personas", nargs="+", default=PERSONAS)
    parser.add_argument("--output_path", type=str,
        default="full_trait_output/plots/persona_layer_dynamics.pdf")
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text())
    layers = payload["layers"]
    results = payload["results"]

    # sized/fonted for a half-linewidth panel (paired with the dot plot)
    fig, ax = plt.subplots(figsize=(4.7, 3.25), dpi=200)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_M, labelsize=10, length=0)

    ax.axhline(0.5, ls=(0, (6, 4)), color="#8A8A8A", lw=1.2, zorder=1)
    ax.axvline(DEFAULT_LAYER, color="#9A9A9A", lw=1.2, ls=(0, (1, 2)), zorder=1)

    for name, (color, marker) in zip(args.personas, SERIES):
        ax.plot(layers, results[name], color=color, lw=2.0, marker=marker,
                ms=3.8, markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.4, label=name, zorder=3)

    t = ax.text(0.985, 0.504, "chance", color=TEXT_M, fontsize=9.5,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    t = ax.text(DEFAULT_LAYER, 1.01, f"layer {DEFAULT_LAYER}", color=TEXT_M,
                fontsize=9.5, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlim(layers[0] - 0.6, layers[-1] + 0.6)
    ax.set_ylim(0.475, 0.92)
    ax.set_xlabel("Layer", fontsize=11, color=TEXT_M)
    ax.set_ylabel("Jailbreak Detection AUC", fontsize=11, color=TEXT_M)

    leg = ax.legend(fontsize=10, loc="upper left", framealpha=0.9,
                    edgecolor=SPINE, borderpad=0.6, handlelength=2.2)
    leg.get_frame().set_facecolor("white")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
