#!/usr/bin/env python3
"""Companion to fig_steering_panels: directional ablation on the initially-complied
pool. Two panels (HarmBench, WildJB), one shared y axis carrying the same rows in
the same order and colours as the steering figure, plus the two pooled random
controls. The x axis is the share of initially-complied prompts the model now
REFUSES; one dot per arm with the 95% bootstrap CI as a soft trace behind it.

Each panel carries its no-intervention line: the refused share of the `none` arm,
i.e. prompts that flip on plain greedy regeneration plus judge noise. Everything
to the right of it is what removing the direction did.

Numbers come from results/ablation_cells.json (extract_ablation_cells.py), scored by
the GPT-4.1-mini judge that produced the pool labels, so the `none` line is the
pool's own reproducibility floor. Degenerate shares are in the appendix table, not
drawn.

Palette is the steering figure's: navy (safety-relevant directions), moss
(individual traits), slate (controls) -- chosen there to survive all three
dichromacies and kept here so the two figures read as one pair.
"""
import argparse
import itertools
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_result_figs import GRID, SPINE, TEXT_M, halo

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "full_trait_output/plots/paper_figs"

C_VEC, C_TRAIT, C_CTRL = "#004F8B", "#448111", "#8A939B"
C_BASE_LINE = "#6E6E6E"
GROUP = {"vec": ("Safety-Relevant Directions", C_VEC),
         "trait": ("Individual Traits", C_TRAIT),
         "ctrl": ("Random Controls", C_CTRL)}
# (label, key, group) -- same order as fig_steering_panels, controls appended
ROWS = [
    ("PersonaJB", "persona", "vec"),
    ("Refusal", "refusal", "vec"),
    ("Jailbreak", "dmu", "vec"),
    ("Utilitarian", "utilitarian", "trait"),
    ("Closure Seeking", "closure_seeking", "trait"),
    ("Edgy", "edgy", "trait"),
    ("Random (full space)", "random_full", "ctrl"),
    ("Random (persona space)", "random_persona", "ctrl"),
]
PANELS = [("HarmBench", "harmbench"), ("WildJB", "wildjb")]
BOLD = {"PersonaJB"}


def pct(x):
    return int(Decimal(repr(round(x, 6))).quantize(Decimal("1"), ROUND_HALF_UP))


def overlaps(fig, texts, pad_pt=0.8):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    p = pad_pt * fig.dpi / 72.0
    b = []
    for t in texts:
        e = t.get_window_extent(r)
        b.append(Bbox.from_extents(e.x0 - p, e.y0 - p, e.x1 + p, e.y1 + p))
    return [(texts[i].get_text(), texts[j].get_text())
            for i, j in itertools.combinations(range(len(texts)), 2)
            if b[i].overlaps(b[j])]


def load_cells(path, placement=""):
    cells = json.load(open(path))

    def get(key, bench):
        # `none` has no hook, so one baseline serves every placement
        suffix = "" if key == "none" else placement
        c = cells.get(f"{key}{suffix}|{bench}")
        if c is None or "refused" not in c:
            return None
        r = c["refused"]
        return r["rate"] * 100, r["lo"] * 100, r["hi"] * 100, c["n"]
    return get


def build(name, get, figw=5.40, height=3.0, gap=0.7, left=0.30, legy=-0.035,
          trace_lw=5.0, trace_alpha=0.22, label_room=9.0):
    rows = [r for r in ROWS if any(get(r[1], b) for _, b in PANELS)]
    ys, bump, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r[2] != prev:
            bump += gap
        ys.append(bump)
        bump += 1.0
        prev = r[2]
    ys = np.array(ys)
    ys = ys.max() - ys

    bases = {b: get("none", b) for _, b in PANELS}
    spans = []
    for _, b in PANELS:
        vals = [get(r[1], b) for r in rows]
        top = max([v[2] for v in vals if v] + [bases[b][2] if bases[b] else 0.0])
        spans.append((0.0, top + label_room))
    fig, axes = plt.subplots(1, 2, figsize=(figw, height), dpi=220, sharey=True,
                             gridspec_kw={"wspace": 0.075,
                                          "width_ratios": [hi - lo for lo, hi in spans]})
    fig.subplots_adjust(left=left, right=0.995, bottom=0.16, top=0.905)
    texts, n_missing = [], 0
    for k, (ax, (title, bkey)) in enumerate(zip(axes, PANELS)):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(colors=TEXT_M, labelsize=7.6, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.7)

        base = bases[bkey]
        if base:
            ax.axvspan(0.0, base[0], color="#F4F4F4", zorder=0)
            ax.axvline(base[0], color=C_BASE_LINE, lw=0.9, ls=(0, (2.2, 1.8)),
                       zorder=2, clip_on=False)
        for y, (lab, key, grp) in zip(ys, rows):
            ax.plot([0.0, spans[k][1]], [y, y], color="#F1F1F1", lw=0.7, zorder=1)
            c = GROUP[grp][1]
            v = get(key, bkey)
            if v is None:
                n_missing += 1
                continue
            x, lo, hi, _ = v
            ax.plot([lo, hi], [y, y], color=c, lw=trace_lw, alpha=trace_alpha,
                    solid_capstyle="round", zorder=3)
            ax.scatter([x], [y], s=17, color=c, edgecolor="white", linewidth=0.5,
                       zorder=5)
            t = ax.text(hi + 2.2, y, f"{pct(x)}", va="center", ha="left",
                        fontsize=6.4, color=TEXT_M, zorder=6,
                        fontweight="bold" if lab in BOLD else "normal")
            halo(t, 2.0)
            texts.append(t)
        for a, b in zip(rows, rows[1:]):
            if a[2] != b[2]:
                yy = (ys[rows.index(a)] + ys[rows.index(b)]) / 2
                ax.axhline(yy, color="#DCDCDC", lw=0.7, zorder=1)
        # the no-intervention line is named once, on the control rows of the
        # first panel, where the band it creates is emptiest
        if k == 0 and base:
            i_ctrl = [i for i, rr in enumerate(rows) if rr[2] == "ctrl"]
            yy = ys[i_ctrl[0]] if i_ctrl else ys[-1]
            t = ax.text(base[0] + 1.6, yy + 0.55, "no intervention", ha="left",
                        va="center", fontsize=6.0, color="#7C7C7C", zorder=6)
            halo(t, 2.5)
            texts.append(t)
        ax.set_xlim(*spans[k])
        ax.set_xticks([t for t in (0, 20, 40, 60, 80, 100) if t <= spans[k][1] - 4])
        ax.set_title(title, fontsize=8.6, color=TEXT_M, pad=6)

    ax0 = axes[0]
    ax0.set_yticks(ys)
    labs = ax0.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    for l, r in zip(labs, rows):
        l.set_linespacing(0.92)
        l.set_color(GROUP[r[2]][1] if r[2] != "ctrl" else TEXT_M)
        if r[0] in BOLD:
            l.set_fontweight("bold")
    ax0.set_ylim(-0.7, ys.max() + 0.7)
    fig.text((left + 0.995) / 2, 0.075,
             "Initially complied prompts now refused (%)", ha="center",
             fontsize=8.8, color=TEXT_M)

    handles = [Line2D([0], [0], marker="o", ls="", color=c, markersize=4.6,
                      markeredgecolor="white", markeredgewidth=0.5, label=n)
               for n, c in GROUP.values()]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.9,
                     framealpha=0.95, edgecolor=SPINE, borderpad=0.35,
                     handletextpad=0.3, columnspacing=1.0, handlelength=1.4,
                     bbox_to_anchor=((left + 0.995) / 2, legy))
    leg.get_frame().set_facecolor("white")

    hits = overlaps(fig, texts + list(labs))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {name:20s} missing={n_missing}  "
          f"span=[{' '.join(f'{a:.0f}-{b:.0f}' for a, b in spans)}]  "
          + (f"OVERLAPS {len(hits)}: {hits[:3]}" if hits else "clean"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(REPO / "results/ablation_cells.json"))
    ap.add_argument("--placement", default="", choices=["", "@16"],
                    help="'' = all layers (Arditi); @16 = that block's output only")
    ap.add_argument("--name", default="fig_ablation_panels")
    ap.add_argument("--out", default=None, help="output dir (default plots/paper_figs)")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    print("building:")
    build(args.name, load_cells(args.cells, args.placement))
