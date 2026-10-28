#!/usr/bin/env python3
"""Steering dose-response, one panel per benchmark: steering coefficient alpha
on x, share of originally-refused prompts flipped to a jailbreak on y. Because
the coefficient sweep stops at the first coefficient that flips a pair, a pair's
min_alpha_to_jailbreak is the smallest coefficient that flips it, and the curve
is the cumulative fraction of pairs flipped at coefficient <= alpha.

Three panels (HarmBench, JBB, WildJB) sharing a y axis and one legend, the same
three benchmarks and six steering arms as fig_steering_panels, in the same
colours (navy = safety-relevant directions, moss = individual traits). At the
alpha = 0.5 budget each curve equals that arm's cell in fig_steering_panels,
which is the cross-check that the extraction is right:
  HarmBench originally-refused pairs (the started_jailbroken flag in the runs),
  JBB all 100 raw behaviours (baseline complies on 1),
  WildJB restricted here to the 975 the unsteered model refused, by joining the
  per-pair rows to results (wildjailbreak_activations/classified_responses.jsonl)
  on behavior_id -- the same restriction fig_steering_panels reads from
  results/wjb_refused_cells.json, reproduced so the panel matches.

Input is results/steering_alpha_curves.json from the jumphost extractor
extract_steering_alpha_curves.py. Bands are 95% bootstrap CIs; the resampling
unit matches steering_cell_cis.json -- behaviours on HarmBench (pairs are
clustered within a behaviour), prompts on JBB and WildJB.

Usage: python3.13 full_trait_tools/plot_steering_alpha_curve.py \
           [--cap 0.5] [--figw 5.51] [--figh 2.0] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_result_figs import GRID, SPINE, TEXT_M

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "full_trait_output/plots/paper_figs"
WJB_LABELS = REPO / "full_trait_output/wildjailbreak_activations/classified_responses.jsonl"
C_VEC, C_TRAIT = "#004F8B", "#448111"     # as in plot_steering_panels
ARMS = [  # key, label, colour, linestyle, bold
    ("persona", "PersonaJB", C_VEC, "-", True),
    ("refusal", "Refusal", C_VEC, "--", False),
    ("dmu", "Jailbreak", C_VEC, ":", False),
    ("utilitarian", "Utilitarian", C_TRAIT, "-", False),
    ("closure_seeking", "Closure Seeking", C_TRAIT, "--", False),
    ("edgy", "Edgy", C_TRAIT, ":", False),
]
PANELS = ["harmbench", "jbb", "wildjb"]
TITLE = {"harmbench": "HarmBench", "jbb": "JBB", "wildjb": "WildJB"}
N_BOOT, SEED = 10_000, 0


def wjb_refused_ids():
    return {json.loads(l)["behavior_id"]
            for l in open(WJB_LABELS) if not json.loads(l)["jailbroken"]}


def by_cluster(cell, bench, refused):
    """cluster id -> array of per-pair min alpha (inf if never flipped)."""
    d = {}
    for cl, a in cell["pairs"]:
        if bench == "wildjb" and cl not in refused:
            continue
        d.setdefault(cl, []).append(np.inf if a is None else float(a))
    return {k: np.array(v) for k, v in d.items()}


def flipped(clusters, grid):
    a = np.concatenate(list(clusters.values()))
    return np.array([(np.isfinite(a) & (a <= g + 1e-9)).mean() for g in grid])


def boot_band(clusters, grid, rng):
    keys = list(clusters)
    stack = np.empty((N_BOOT, len(grid)))
    for b in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        a = np.concatenate([clusters[keys[i]] for i in pick])
        stack[b] = [(np.isfinite(a) & (a <= g + 1e-9)).mean() for g in grid]
    return np.percentile(stack, [2.5, 97.5], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "results/steering_alpha_curves.json"))
    p.add_argument("--cap", type=float, default=0.5)
    p.add_argument("--figw", type=float, default=5.51)   # \linewidth = 397pt
    p.add_argument("--figh", type=float, default=2.0)
    p.add_argument("--no_band", action="store_true")
    p.add_argument("--out", default=str(OUT / "fig_steering_alpha.pdf"))
    args = p.parse_args()

    data = json.load(open(args.data))
    refused = wjb_refused_ids()
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(args.figw, args.figh),
                             sharey=True)
    ymax = 0
    for ax, bench in zip(axes, PANELS):
        print(TITLE[bench])
        for key, label, color, ls, bold in ARMS:
            cell = data.get(f"{key}|{bench}")
            if cell is None:
                print(f"  {key}: missing, skipped")
                continue
            clusters = by_cluster(cell, bench, refused)
            sched = sorted(a for a in (cell.get("schedule") or [])
                           if a <= args.cap + 1e-9)
            grid = np.array([0.0] + sched)
            y = 100 * flipped(clusters, grid)
            if not args.no_band:
                lo, hi = 100 * boot_band(clusters, grid, rng)
                ax.fill_between(grid, lo, hi, color=color, alpha=0.13, lw=0)
                ymax = max(ymax, hi.max())
            ax.plot(grid, y, color=color, ls=ls, lw=1.6 if bold else 1.1,
                    solid_capstyle="round", marker="o",
                    ms=3.0 if bold else 2.4, markeredgewidth=0)
            n = sum(len(v) for v in clusters.values())
            print(f"  {label:16s} n={n:5d}  flipped@{args.cap:.2f} = {y[-1]:.1f}%")
        ax.set_title(TITLE[bench], fontsize=9, color=TEXT_M)
        ax.set_xlim(0, args.cap)
        ax.set_xticks(sched)
        ax.tick_params(labelsize=7.5, colors=TEXT_M)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(SPINE)
    axes[len(PANELS) // 2].set_xlabel("Steering coefficient", fontsize=8.5,
                                      color=TEXT_M)
    axes[0].set_ylim(0, ymax * 1.04)
    axes[0].set_ylabel("Refused prompts\nflipped (%)", fontsize=8.5, color=TEXT_M)

    handles = [Line2D([0], [0], color=c, ls=ls, lw=1.6 if b else 1.1,
                      marker="o", ms=3.0 if b else 2.4, markeredgewidth=0,
                      label=lab) for _, lab, c, ls, b in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
               frameon=False, handlelength=1.8, columnspacing=1.1,
               handletextpad=0.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.4, w_pad=0.8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
