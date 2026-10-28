#!/usr/bin/env python3
"""
plot_trait_auc_dotplot.py

Paper figure replacing the trait-AUC tables: dot plot of the most predictive
individual trait directions for both models, split vertically into a
Llama-3.1-8B block (top) and a Gemma-4-31B block (bottom).

Dot: AUC averaged over the 101 held-out pool splits.
Whisker: 95% CI of that mean (1.96 * std / sqrt(n_seeds)).
Vertical lines: chance (0.5) and, per block, the AUC averaged over all of
that model's trait directions.

Llama: HarmBench, layer 16. Gemma: HarmBench+WildJailbreak (evalmerge),
layer 30 — same sources as the appendix tables.

Usage:
  python3 full_trait_tools/plot_trait_auc_dotplot.py \
      [--llama_csv results/persona_individual_auc_seeds0-100.csv] \
      [--gemma_json full_trait_output/persona_individual_auc_gemma_evalmerge/results.json] \
      [--output_path full_trait_output/plots/trait_auc_dotplot.pdf]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

N_TOP = 5

# journal palette; llama brick matches the persona-arrow color in the PCA
# panel, gemma amber is the CVD-checked companion from the per-layer figure
C_LLAMA = "#A6403A"
C_GEMMA = "#C88A2D"
GRID    = "#E7E7E7"
SPINE   = "#C9C9C9"
TEXT_M  = "#555555"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_llama(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: -float(r["mean_auc"]))
    return [(r["trait"], float(r["mean_auc"]), float(r["std_auc"]),
             int(r["n_seeds"])) for r in rows]


def load_gemma(path):
    d = json.loads(Path(path).read_text())
    return [(r["trait"], float(r["mean_auc"]), float(r["std_auc"]),
             int(r["n_seeds"])) for r in d["ranked"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_csv", type=str,
        default="results/persona_individual_auc_seeds0-100.csv")
    parser.add_argument("--gemma_json", type=str,
        default="full_trait_output/persona_individual_auc_gemma_evalmerge/results.json")
    parser.add_argument("--output_path", type=str,
        default="full_trait_output/plots/trait_auc_dotplot.pdf")
    args = parser.parse_args()

    blocks = [
        ("Llama-3.1-8B", C_LLAMA, load_llama(args.llama_csv)),
        ("Gemma-4-31B", C_GEMMA, load_gemma(args.gemma_json)),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.15), dpi=200)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_M, labelsize=10, length=0)
    ax.tick_params(axis="y", labelleft=False)

    def halo(t, lw=3):
        t.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])

    y = 0.0
    all_ys = []
    for model, color, traits in blocks:
        t = ax.text(0.452, y, model, color=color, fontsize=10.5,
                    fontweight="bold", style="italic", ha="left", va="center")
        halo(t)
        y -= 1.0
        block_top = y + 0.5
        for name, v, sd, n in traits[:N_TOP]:
            ci = 1.96 * sd / np.sqrt(n)
            ax.plot([v - ci, v + ci], [y, y], color=color, lw=2.4,
                    solid_capstyle="round", zorder=4)
            ax.scatter([v], [y], color=color, s=34, zorder=5)
            t = ax.text(v + ci + 0.010, y, name, color=color, fontsize=10.5,
                        fontweight="bold", ha="left", va="center", zorder=6)
            halo(t)
            all_ys.append(y)
            y -= 1.0
        # per-model average over all of its trait directions
        avg = float(np.mean([v for _, v, _, _ in traits]))
        ax.plot([avg, avg], [block_top, y + 0.5], color="#9A9A9A", lw=1.2,
                zorder=1)
        t = ax.text(avg, y + 0.35, "avg trait", color=TEXT_M, fontsize=9,
                    ha="center", va="top")
        halo(t)
        y -= 1.6  # gap before next block

    y_bot = y - 0.4
    ax.axvline(0.5, ls=(0, (6, 4)), color="#8A8A8A", lw=1.2, zorder=1)
    t = ax.text(0.5, y_bot + 0.15, "chance", color=TEXT_M, fontsize=9.5,
                ha="center", va="bottom")
    halo(t)

    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(0.445, 1.06)
    ax.set_ylim(y_bot, 0.7)
    ax.set_xlabel("Jailbreak Detection AUC", fontsize=11, color=TEXT_M)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
