#!/usr/bin/env python3
"""
Gemma-4-31B steering dose-response, one panel per benchmark: steering coefficient alpha on x,
share of originally-refused prompts flipped on y. Counterpart of plot_steering_alpha_curve.py,
but it reads the per-pair result files directly (full_trait_output/outputs_gemma_steer/
<bench>/<arm>/results.jsonl) instead of a jumphost-extracted curve file -- every record already
carries min_alpha_to_jailbreak, behavior_id and (on WildJB) baseline_jailbroken.

Because a pair drops out of the escalation at the first alpha that flips it, a pair's
min_alpha_to_jailbreak is the smallest coefficient that flips it and the curve is the cumulative
fraction flipped at coefficient <= alpha. At the schedule ceiling each curve equals that arm's
cell in fig_steering_panels_gemma, which is the cross-check that the extraction is right.

Only arms whose run finished (summary.json present) are drawn: while an arm is running its
results.jsonl holds ONLY the pairs that already flipped, so a partial file reads as 100%.
Arms extended past the original ceiling (--extend_schedule) are handled: the grid is the union
of alphas present, and an arm that was never extended stops at its own last alpha.

Bands are 95% bootstrap CIs, resampling behaviours on HarmBench (pairs are clustered within a
behaviour) and prompts on JBB and WildJB.

Usage: python3 full_trait_tools/plot_gemma_steering_alpha_curve.py [--traits 3] [--cap 0.5]
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
from extract_gemma_steering_cells import looks_like_text   # same gibberish guard as the cells

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "full_trait_output/outputs_gemma_steer"
OUT = REPO / "full_trait_output/plots/paper_figs_gemma"
C_VEC, C_TRAIT = "#004F8B", "#448111"
VEC_ARMS = [("w_persona__full", "PersonaJB", "-", True),
            ("refusal_pos1_neg__full", "Refusal", "--", False),
            ("dmu__full", "Jailbreak", ":", False)]
TRAIT_STYLES = ["-", "--", ":", "-."]
PANELS = ["harmbench", "jbb", "wildjb"]
TITLE = {"harmbench": "HarmBench", "jbb": "JBB", "wildjb": "WildJB"}
N_BOOT, SEED = 10_000, 0


def load_arm(bench, arm):
    """-> (clusters {id: array of min alpha, inf = never flipped}, alphas tried) or None."""
    d = ROOT / bench / arm
    if not (d / "summary.json").exists() or not (d / "results.jsonl").exists():
        return None
    rows = []
    for line in open(d / "results.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # Apply the same character-level gibberish guard the cell extractor uses, so the curves and the
    # table cannot disagree: a "jailbreak" whose response is not prose is a failure, not a success.
    for r in rows:
        if r.get("final_jailbroken") and not looks_like_text(r.get("final_response", "")):
            r["final_jailbroken"] = False
            r["min_alpha_to_jailbreak"] = None
    if bench == "wildjb":                      # fig:attack convention: refused-only
        rows = [r for r in rows if not r.get("baseline_jailbroken")]
    clusters, alphas = {}, set()
    for r in rows:
        a = r.get("min_alpha_to_jailbreak")
        clusters.setdefault(str(r["behavior_id"]), []).append(np.inf if a is None else float(a))
        if a is not None:
            alphas.add(round(float(a), 3))
        if r.get("max_alpha_tried"):
            alphas.add(round(float(r["max_alpha_tried"]), 3))
    return {k: np.asarray(v) for k, v in clusters.items()}, sorted(alphas)


def curve(clusters, grid):
    a = np.concatenate(list(clusters.values()))
    return np.array([(np.isfinite(a) & (a <= g + 1e-9)).mean() for g in grid])


def band(clusters, grid, rng):
    keys = list(clusters)
    stack = np.empty((N_BOOT, len(grid)))
    for b in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        a = np.concatenate([clusters[keys[i]] for i in pick])
        stack[b] = [(np.isfinite(a) & (a <= g + 1e-9)).mean() for g in grid]
    return np.percentile(stack, [2.5, 97.5], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traits", type=int, default=3,
                   help="how many single-trait arms to draw (strongest by final HarmBench flip rate)")
    p.add_argument("--cap", type=float, default=0.0, help="0 = every alpha present in the data")
    p.add_argument("--figw", type=float, default=5.51)
    p.add_argument("--figh", type=float, default=2.0)
    p.add_argument("--no_band", action="store_true")
    p.add_argument("--out", default=str(OUT / "fig_steering_alpha_gemma.pdf"))
    a = p.parse_args()

    # rank the trait arms on HarmBench, so the figure shows the strongest ones
    ranked = []
    for d in sorted((ROOT / "harmbench").glob("trait__*")) if (ROOT / "harmbench").exists() else []:
        got = load_arm("harmbench", d.name)
        if got:
            cl, al = got
            ranked.append((curve(cl, [max(al)])[0], d.name))
    ranked.sort(reverse=True)
    arms = [(k, lab, C_VEC, ls, bold) for k, lab, ls, bold in VEC_ARMS]
    arms += [(name, name.replace("trait__", "").replace("_", " ").title(), C_TRAIT,
              TRAIT_STYLES[i % len(TRAIT_STYLES)], False)
             for i, (_, name) in enumerate(ranked[:a.traits])]
    if ranked:
        print("trait arms by HarmBench flip rate: " +
              ", ".join(f"{n.replace('trait__','')} {v:.0%}" for v, n in ranked))

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(a.figw, a.figh), sharey=True)
    ymax, drawn, xticks = 0.0, 0, []
    for ax, bench in zip(axes, PANELS):
        print(TITLE[bench])
        for key, label, color, ls, bold in arms:
            got = load_arm(bench, key)
            if got is None:
                print(f"  {label:16s} not finished, skipped")
                continue
            clusters, alphas = got
            sched = [x for x in alphas if not a.cap or x <= a.cap + 1e-9]
            if not sched:
                continue
            grid = np.array([0.0] + sched)
            xticks = sorted(set(xticks) | set(sched))
            y = 100 * curve(clusters, grid)
            if not a.no_band:
                lo, hi = 100 * band(clusters, grid, rng)
                ax.fill_between(grid, lo, hi, color=color, alpha=0.13, lw=0)
                ymax = max(ymax, hi.max())
            ymax = max(ymax, y.max())
            ax.plot(grid, y, color=color, ls=ls, lw=1.6 if bold else 1.1,
                    solid_capstyle="round", marker="o", ms=3.0 if bold else 2.4,
                    markeredgewidth=0)
            drawn += 1
            n = sum(len(v) for v in clusters.values())
            print(f"  {label:16s} n={n:5d}  flipped@{max(sched):.2f} = {y[-1]:.1f}%")
        ax.set_title(TITLE[bench], fontsize=9, color=TEXT_M)
        ax.tick_params(labelsize=7.5, colors=TEXT_M)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(SPINE)
    if not drawn:
        print("nothing finished yet -- no figure written")
        return
    for ax in axes:
        ax.set_xlim(0, max(xticks))
        ax.set_xticks(xticks)
    axes[len(PANELS) // 2].set_xlabel("Steering coefficient", fontsize=8.5, color=TEXT_M)
    axes[0].set_ylim(0, max(ymax * 1.04, 1.0))
    axes[0].set_ylabel("Refused prompts\nflipped (%)", fontsize=8.5, color=TEXT_M)
    handles = [Line2D([0], [0], color=c, ls=ls, lw=1.6 if b else 1.1, marker="o",
                      ms=3.0 if b else 2.4, markeredgewidth=0, label=lab)
               for _, lab, c, ls, b in arms]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=7,
               frameon=False, handlelength=1.8, columnspacing=1.1, handletextpad=0.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.4, w_pad=0.8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(str(a.out).replace(".pdf", ".png"), dpi=220)
    print("saved", a.out)


if __name__ == "__main__":
    main()
