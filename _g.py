#!/usr/bin/env python3
"""Three-panel dot plot for Tables 4 and 5, ASR in percent, with bootstrap CIs.

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
OUT = Path(r"/private/tmp/claude-503/-Users-viktor/f5bec5b3-dfb4-4635-aead-22d9431d3c4c/scratchpad/tabfig")
OUT.mkdir(parents=True, exist_ok=True)
CIS = json.load(open(REPO / "results/steering_cell_cis.json"))

# Deliberately NOT brick/amber: those are the paper's Llama/Gemma pair and
# would read as a model split here. Navy + moss + slate is the pairing that
# survives all three dichromacies -- blue against green collapses under
# tritanopia, and red-orange and pink are ruled out, which leaves a cool hue
# against a yellow-green one. Chroma is ~0.15, above the paper's usual 0.08-
# 0.14, because a muted olive at this lightness reads as mud.
C_VEC, C_TRAIT = "#004F8B", "#448111"
C_STD = "#8A939B"     # neutral for the baseline group, as elsewhere in the paper
C_BASE_LINE = "#6E6E6E"
# wrapped by hand: a rotated group name has only its own block's height to
# live in, and "Safety-Relevant Directions" does not fit across three rows
GROUP = {"vec": ("Safety-Relevant Directions", C_VEC),
         "trait": ("Individual Traits", C_TRAIT),
         "std": ("Standard Attacks", C_STD)}

# (label, key, HarmBench, JBB, WildJB, group) in percent -- Tables 4 and 5
# verbatim, EXCEPT PAIR and PAP on HarmBench: the table prints 39 and 28, which
# no combination of judge field, category subset, behaviour subset or basis in
# the stored data reproduces, so both are corrected to the per-pair value the
# data gives (43.25 and 26.40) -- the same basis as the other three families in
# that column. Point and interval then always come from one source.
ROWS = [
    ("Jailbreak Persona\nDirection",    "persona",         73.6, 58.0, 77.3, "vec"),
    ("Refusal Direction",              "refusal",         72.7, 82.0, 75.0, "vec"),
    ("Jailbreak Direction",            "dmu",             67.6, 59.0, 81.5, "vec"),
    ("Utilitarian",                    "utilitarian",     58.8, 69.0, 75.7, "trait"),
    ("Closure Seeking",                "closure_seeking", 55.0, 55.0, 80.0, "trait"),
    ("Edgy",                           "edgy",            50.7, 41.0, 84.3, "trait"),
    ("GPTFuzz",                        "GPTFuzz",         59.0, 77.0, 54.0, "std"),
    ("GCG",                            "GCG",             44.0, 40.0, 38.0, "std"),
    ("PAIR",                           "PAIR",            43.25, 28.0, 22.0, "std"),
    ("PAP",                            "PAP",             26.40, 21.0, 50.0, "std"),
    ("PEZ",                            "PEZ",             15.0, 2.0, 42.0, "std"),
]
# (display name, cluster key, unsteered rate in percent)
PANELS = [("HarmBench", "harmbench", 0.0),
          ("JBB", "jbb", 1.0),
          ("WildJB", "wildjb", 51.2)]
BOLD = {"Jailbreak Persona\nDirection"}
TOL = 0.51        # percent; values are printed to 2-3 dp as fractions


def pct(x):
    """Integer percent, rounded half-up (43.25 -> 43, 41.5 -> 42)."""
    return int(Decimal(repr(round(x, 6))).quantize(Decimal("1"), ROUND_HALF_UP))


def cell(key, bench, table_value):
    """-> (x, lo, hi, verified), all in percent."""
    d = CIS.get(f"{key}|{bench}")
    if d is None or abs(d["asr"] * 100 - table_value) > TOL:
        return table_value, None, None, False
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
        vals = [cell(r[1], key, r[2 + k]) for r in rows]
        top = max((v[2] or v[0]) for v in vals)
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


def build(name, trace_lw=5.0, trace_alpha=0.22, figw=5.40, height=3.05,
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
    # past the canvas: the row labels are long enough ("Jailbreak Persona
    # Direction") that the overflow pushed the saved figure to 441pt
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
            x, lo, hi, ok = cell(key, bkey, v[k])
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
        if base > 20:
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


print("building:")
# height/font chosen by measurement: the first pair at which every rotated
# group name fits inside its own block (3.05/6.8 overflows the 3-row blocks)
# 3.35in: the first height at which the wrapped label clears its
# neighbour by more than 3pt once its leading is tightened
# legy chosen as the tightest position that still clears the axis label
build("G0.020", height=3.35, gap=0.7, left=0.215,
      legy=-0.02)
