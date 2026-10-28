#!/usr/bin/env python3
"""GEMMA-4-31B variant (generated from plot_steering_panels.py; data section replaced).
Three-panel dot plot for Tables 4 and 5, ASR in percent, with bootstrap CIs.

One panel per dataset, one shared y axis carrying the steering vectors, the
single traits and the standard attacks, one dot per row for where that attack
landed, and a soft trace behind it for the 95% CI.

No legend: each group is named down the left margin next to its own rows, in
its own colour, so the colour key sits where the rows are instead of in a box
the reader has to look away to.

Each panel is cut to the range it actually needs, but the panel *widths* are
proportional to those ranges, so one percentage point is the same physical
distance in all three -- the dead space on the right goes away without making
the panels incomparable.

Every panel also carries its own no-intervention line: the rate the unsteered
model already achieves on that set (0% on HarmBench's originally-refused group,
1% on JBB, 51% on WildJB). It is labelled in the WildJB panel, the only one
where it sits away from the edge and where attacks actually fall below it.

CIs come from compute_cis.py, which bootstraps the per-prompt outcomes stored
on the cluster: behaviours resampled where pairs are clustered inside them
(HarmBench: 1935 pairs over 159 behaviours), pairs otherwise.

Dropped per request: steer-$w$ (raw), steer-$w$ (aligned), steer-traits, and
all traits except edgy / utilitarian / closure_seeking.
"""
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
from matplotlib.transforms import Bbox, blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_result_figs import GRID, SPINE, TEXT_M, halo

REPO = Path(__file__).resolve().parents[1]
CIS = json.load(open(REPO / "results/gemma_steering_cell_cis.json"))
CIS.update(json.load(open(REPO / "results/gemma_steering_cell_cis_classic.json")))
OUT = REPO / "full_trait_output/plots/paper_figs_gemma"
OUT.mkdir(parents=True, exist_ok=True)

C_VEC, C_TRAIT = "#004F8B", "#448111"
C_STD = "#8A939B"
C_BASE_LINE = "#6E6E6E"
GROUP = {"vec": ("Safety-Relevant Directions", C_VEC),
         "trait": ("Individual Traits", C_TRAIT),
         "std": ("Standard Attacks", C_STD)}

# Gemma-4-31B, layer 30. Every value comes from results/gemma_steering_cell_cis.json
# (extract_gemma_steering_cells.py over outputs_gemma_steer/) -- there is no separate table
# to cross-check against yet, so the "table value" IS the cell value. WildJB uses the
# refused-only cells (fig:attack convention). Standard attacks exist for Gemma on HarmBench
# only (the detection attack dumps); their JBB/WildJB cells are absent and skipped.
STEER = [("PersonaJB", "w_persona__full", "vec"),
         ("Refusal", "refusal_pos1_neg__full", "vec"),
         ("Jailbreak", "dmu__full", "vec")]
STEER += [(t.capitalize(), f"trait__{t}", "trait") for t in
          ["romantic", "stoic", "dispassionate", "emotional", "literal",
           "flirty", "patient", "dominant", "reserved", "confident"]]
STD = [(a, a, "std") for a in ["GPTFuzz", "GCG", "PAIR", "PAP", "PEZ"]]
BENCH_KEYS = {"harmbench": "harmbench", "jbb": "jbb", "wildjb": "wildjb_refused"}


def _val(key, bench):
    d = CIS.get(f"{key}|{BENCH_KEYS[bench]}")
    return None if d is None else d["asr"] * 100.0


ROWS = [(lab, key, _val(key, "harmbench"), _val(key, "jbb"), _val(key, "wildjb"), grp)
        for lab, key, grp in STEER + STD]
PANELS = [("HarmBench", "harmbench", 0.0),
          ("JBB", "jbb", 0.0),
          ("WildJB", "wildjb_refused", 0.0)]
BOLD = {"PersonaJB"}
TOL = 0.51


def pct(x):
    """Integer percent, rounded half-up (43.25 -> 43, 41.5 -> 42)."""
    return int(Decimal(repr(round(x, 6))).quantize(Decimal("1"), ROUND_HALF_UP))


def cell(key, bench, table_value):
    """-> (x, lo, hi, verified), all in percent; None if the cell does not exist."""
    d = CIS.get(f"{key}|{bench}")
    if d is None:
        return None
    return d["asr"] * 100, d["lo"] * 100, d["hi"] * 100, True


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


def panel_ranges(rows, label_room=9.0, cut_left=False):
    """(xmin, xmax) per panel. xmax is the widest CI plus room for the value
    label. xmin is 0 unless cut_left, in which case a panel whose
    no-intervention line is well above 0 may start below its lowest CI instead
    -- the panel keeps its dashed line and the band under it, just not the
    empty stretch down to zero. Widths are made proportional to the ranges, so
    the scale stays shared even though the windows differ."""
    out = []
    for k, (_, key, base) in enumerate(PANELS):
        vals = [c for c in (cell(r[1], key, r[2 + k]) for r in rows) if c is not None]
        top = max((v[2] or v[0]) for v in vals) if vals else 10.0
        bot = 0.0
        if cut_left and base > 20:
            lowest = min((v[1] if v[1] is not None else v[0]) for v in vals)
            bot = max(0.0, 5.0 * np.floor((lowest - 3.0) / 5.0))
        out.append((bot, top + label_room))
    return out


def group_bar(ax, x, y0, y1, transform, color):
    """Plain rule spanning the group's rows."""
    ax.plot([x, x], [y0, y1], transform=transform, color=color, lw=1.6,
            alpha=0.8, solid_capstyle="round", clip_on=False, zorder=3)


def build(name, trace_lw=5.0, trace_alpha=0.22, figw=5.40, height=4.6,
          gap=1.25, left=0.375, legy=-0.115,
          cut_left=False):
    rows = list(ROWS)
    ys, bump, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r[5] != prev:
            bump += gap
        ys.append(bump)
        bump += 1.0
        prev = r[5]
    ys = np.array(ys)
    ys = ys.max() - ys                     # first row at the top

    spans = panel_ranges(rows, cut_left=cut_left)
    xmaxes = [hi for _, hi in spans]
    fig, axes = plt.subplots(1, 3, figsize=(figw, height), dpi=220, sharey=True,
                             gridspec_kw={"wspace": 0.075,
                                          "width_ratios": [hi - lo for lo, hi in spans]})
    # reserve the left margin rather than letting bbox_inches="tight" expand
    # past the canvas: the row labels are long enough ("Closure Seeking")
    # that the overflow once pushed the saved figure to 441pt
    fig.subplots_adjust(left=left, right=0.995, bottom=0.145, top=0.905)
    texts, n_unverified = [], 0
    for k, (ax, (title, bkey, base)) in enumerate(zip(axes, PANELS)):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(colors=TEXT_M, labelsize=7.6, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.7)

        # what the model already does with no intervention; clip_on=False so
        # the HarmBench line, which sits at 0, is not swallowed by the edge
        ax.axvspan(spans[k][0], base, color="#F4F4F4", zorder=0)
        ax.axvline(base, color=C_BASE_LINE, lw=0.9, ls=(0, (2.2, 1.8)),
                   zorder=2, clip_on=False)

        for y, (lab, key, *v, grp) in zip(ys, rows):
            ax.plot([spans[k][0], xmaxes[k]], [y, y], color="#F1F1F1", lw=0.7,
                    zorder=1)
            c = GROUP[grp][1]
            got = cell(key, bkey, v[k])
            if got is None:
                t = ax.text(spans[k][0] + 1.5, y, "n/a", va="center", ha="left", fontsize=5.8,
                            color="#B0B0B0", zorder=6)
                texts.append(t)
                continue
            x, lo, hi, ok = got
            if ok:
                ax.plot([lo, hi], [y, y], color=c, lw=trace_lw,
                        alpha=trace_alpha, solid_capstyle="round", zorder=3)
                ax.scatter([x], [y], s=17, color=c, edgecolor="white",
                           linewidth=0.5, zorder=5)
                right = hi
            else:
                n_unverified += 1
                ax.scatter([x], [y], s=19, facecolor="white", edgecolor=c,
                           linewidth=1.0, zorder=5)
                right = x
            t = ax.text(right + 2.2, y, f"{pct(x)}", va="center", ha="left",
                        fontsize=6.4, color=TEXT_M, zorder=6,
                        fontweight="bold" if lab in BOLD else "normal")
            halo(t, 2.0)
            texts.append(t)

        for a, b in zip(rows, rows[1:]):
            if a[5] != b[5]:
                yy = (ys[rows.index(a)] + ys[rows.index(b)]) / 2
                ax.axhline(yy, color="#DCDCDC", lw=0.7, zorder=1)

        # the dashed line only carries information where it is away from the
        # edge, so it is named once, in the empty band it creates on WildJB
        if False:
            # on the trait rows: every trait lands at 76-84 on WildJB, so the
            # band left of the line is empty there (the GCG row is not)
            mid_trait = [i for i, rr in enumerate(rows) if rr[5] == "trait"][1]
            t = ax.text(base - 2.4, ys[mid_trait], "no intervention",
                        rotation=90, ha="center", va="center", fontsize=6.0,
                        color="#7C7C7C", zorder=6)
            halo(t, 2.5)
            texts.append(t)

        ax.set_xlim(spans[k][0], xmaxes[k])
        ax.set_xticks([t for t in (0, 20, 40, 60, 80)
                       if spans[k][0] + 3 <= t <= xmaxes[k] - 4])
        ax.set_title(title, fontsize=8.6, color=TEXT_M, pad=6)

    ax0 = axes[0]
    fig.suptitle("Gemma-4-31B, layer 30", fontsize=8.6, color=TEXT_M, y=0.985)
    ax0.set_yticks(ys)
    labs = ax0.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    for l, r in zip(labs, rows):
        # the one wrapped label sets the row pitch for the whole figure, so
        # tighten its leading rather than growing every other row to suit it
        l.set_linespacing(0.92)
        l.set_color(GROUP[r[5]][1] if r[5] != "std" else TEXT_M)
        if r[0] in BOLD:
            l.set_fontweight("bold")
    ax0.set_ylim(-0.7, ys.max() + 0.7)
    # left on the middle axis: it lands within 0.3% of the legend's centre
    # anyway, and repositioning it by hand only pushed it into the legend
    axes[1].set_xlabel("Attack Success Rate (%)",
                       fontsize=8.8, color=TEXT_M, labelpad=5)

    handles = [Line2D([0], [0], marker="o", ls="", color=c, markersize=4.6,
                      markeredgecolor="white", markeredgewidth=0.5, label=n)
               for n, c in GROUP.values()]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.9,
                     framealpha=0.95, edgecolor=SPINE, borderpad=0.35,
                     handletextpad=0.3, columnspacing=1.0, handlelength=1.4,
                     bbox_to_anchor=((left + 0.995) / 2, legy))
    leg.get_frame().set_facecolor("white")
    gtexts = []

    hits = overlaps(fig, texts + list(labs))
    # touching is not the bar to clear -- a wrapped label can sit 1pt from its
    # neighbour, pass an overlap test and still read as cramped
    fig.canvas.draw()
    rr_ = fig.canvas.get_renderer()
    boxes = [l.get_window_extent(rr_) for l in labs]
    min_gap_pt = min((boxes[i].y0 - boxes[i + 1].y1) * 72.0 / fig.dpi
                     for i in range(len(boxes) - 1))
    if min_gap_pt < 3.0:
        hits.append((f"<row labels only {min_gap_pt:.1f}pt apart>", ""))
    if leg.get_window_extent(rr_).overlaps(
            axes[1].xaxis.label.get_window_extent(rr_)):
        hits.append(("<legend>", "<xlabel>"))

    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {name:16s} unverified={n_unverified}  " +
          f"span=[{' '.join(f'{a:.0f}-{b:.0f}' for a, b in spans)}]  " +
          (f"OVERLAPS {len(hits)}: {hits[:3]}" if hits else "clean"))


print("building (gemma):")
build("fig_steering_panels_gemma", height=4.6, gap=0.7, left=0.150, legy=-0.03)
