#!/usr/bin/env python3
"""fig_direction_story -- refusal and jailbreak directions live in persona space.

Left: share of each direction's squared norm inside the trait span.
Right: held-out jailbreak-detection AUC of the full direction (circle) and of
its component inside the persona span (star), on a truncated axis, with 95% CI
bands over 50 behaviour- and template-disjoint splits.

Numbers (verified 2026-08-20, evalmerge protocol, raw space): Llama = the
reproducible 50-seed run, Gemma = 50-seed HB+WildJailbreak pool,
hidden_states[31] pos -1. Chance share is span_dim/hidden_dim: Llama
229/4096 = 5.6%, Gemma 240/5376 = 4.5%. NOTE: an earlier version of this file
carried pre-50-seed values (44.6% / AUC 0.818, 0.842) and no CI bands; those
are superseded and must not come back -- the deployed figure and the caption
in sections/interp.tex both describe the values below.

Geometry: the figure is emitted at exactly \\linewidth (397pt) so LaTeX does
not rescale it and shrink every font with it. Vertical extent is as small as
the collision checks allow at unchanged font size; the two in-axes notes are
dropped because the caption already states both facts.

Usage: python3.13 plot_direction_story.py [--variant roomy|notes|floor]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

OUT = Path("full_trait_output/plots/paper_figs")
C_REF, C_DMU = "#1D5B8C", "#C88A2D"
GRID, SPINE, TEXT_M = "#E7E7E7", "#C9C9C9", "#555555"
LINEWIDTH_PT = 397.0
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

# (model, colour, share %, AUC full, AUC in-span, chance share %)
ROWS = [
    ("Llama-3.1-8B", C_REF, 40.7, 0.844, 0.868, 5.6),
    ("Gemma-4-31B",  C_REF, 26.5, 0.799, 0.772, 4.5),
    ("Llama-3.1-8B", C_DMU, 48.2, 0.856, 0.846, 5.6),
    ("Gemma-4-31B",  C_DMU, 33.0, 0.861, 0.857, 4.5),
]
# 95% CI of the mean over 50 splits (1.96*std/sqrt(50)): (full, component)
CIS = {
    ("Llama-3.1-8B", C_REF): (0.0192, 0.0186),
    ("Gemma-4-31B", C_REF): (0.0164, 0.0169),
    ("Llama-3.1-8B", C_DMU): (0.0221, 0.0210),
    ("Gemma-4-31B", C_DMU): (0.0172, 0.0141),
}
HEAD_NAMES = ["Refusal Direction", "Jailbreak Direction"]

# dy = pitch between the two models of one direction; gg = pitch between the
# two direction groups (the header sits in that gap, ho above the group's top
# bar); notes = keep the two in-axes annotations, which cost ~20pt of height.
BASE = dict(dy=0.55, gg=1.00, ho=0.53, top=0.20, bar=0.32,
            band=5.2, circ=28, star=100, tick=0.22)
VARIANTS = {
    "roomy": dict(BASE, figh=1.22, bot=0.18, notes=False),   # shipped
    "notes": dict(BASE, figh=1.56, bot=0.50, notes=True),
    "floor": dict(BASE, figh=1.17, bot=0.18, notes=False),   # collision floor
}
BAR_GAP_PT = 2.5      # white space between two bars; below this they merge
HEAD_CLEAR_PT = 1.5   # header text above the bar it sits over
FRAME_PT = 3.0        # ink to the axes frame, top and bottom


def halo(t, lw=2.5):
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])


def ink(ax, rend):
    """Every drawn thing in `ax` as (name, display-space bbox).

    Not just get_window_extent(): a scatter PathCollection reports (inf, -inf)
    here, so markers and CI bands are invisible to a naive sweep and a clearance
    solver would happily park the top row's star on the frame. Offsets are
    transformed by hand and padded by the marker's own size; band lines are
    padded by half their linewidth. The axis-break marks straddle the spine on
    purpose and are skipped.
    """
    dpi = ax.figure.dpi / 72.0
    out = [(t.get_text().strip()[:20], t.get_window_extent(rend))
           for t in ax.texts if t.get_text().strip()]
    out += [("bar", p.get_window_extent(rend)) for p in ax.patches]
    for c in ax.collections:
        pad = (max(c.get_sizes()) ** 0.5 / 2 + 1.0) * dpi
        for x, y in ax.transData.transform(c.get_offsets()):
            out.append(("marker", Bbox([[x - pad, y - pad], [x + pad, y + pad]])))
    for ln in ax.lines:
        if ln.get_transform() is not ax.transData:
            continue                       # axis-break marks sit on the spine
        pad = ln.get_linewidth() / 2 * dpi
        out.append(("band", ln.get_window_extent(rend).padded(pad)))
    leg = ax.get_legend()
    if leg is not None:
        out.append(("legend", leg.get_window_extent(rend)))
    return out


def style(ax, f):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_M, labelsize=8.5, length=0, pad=2)


def build(g, figw):
    dy, gg, ho = g["dy"], g["gg"], g["ho"]
    ys = [2 * dy + gg, dy + gg, dy, 0.0]          # top row first, as ROWS
    heads = [ys[0] + ho, ys[2] + ho]
    ylim = (-g["bot"], heads[0] + g["top"])

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(figw, g["figh"]), dpi=200, sharey=True,
        gridspec_kw={"width_ratios": [1.0, 0.62], "wspace": 0.10})

    # ---- left: share of norm inside the trait span ----
    style(axL, g)
    axL.grid(axis="x", color=GRID, linewidth=0.8)
    for y, (model, c, share, full, proj, chance) in zip(ys, ROWS):
        axL.barh(y, share, height=g["bar"], color=c, alpha=0.85, zorder=3)
        t = axL.text(share + 1.3, y, f"{share:.1f}%", va="center",
                     fontsize=8.5, color=c, fontweight="bold")
        halo(t)
        axL.plot([chance, chance], [y - g["tick"], y + g["tick"]],
                 ls=(0, (3, 2)), color="#8A8A8A", lw=1.0, zorder=4)
    if g["notes"]:
        t = axL.text(7.0, ylim[0] + 0.03, "random direction (chance)",
                     fontsize=7.2, color=TEXT_M, ha="left", va="bottom")
        halo(t)
    axL.set_yticks(ys)
    axL.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
    for name, c, yh in zip(HEAD_NAMES, (C_REF, C_DMU), heads):
        t = axL.text(0.8, yh, name, color=c, fontsize=9, fontweight="bold",
                     ha="left", va="center", zorder=6)
        halo(t, 3)
    axL.set_xlim(0, 62)          # 62 not 56: the 48.2% label ends at 58.0
    axL.set_xticks([0, 10, 20, 30, 40, 50])
    axL.set_ylim(*ylim)
    axL.set_xlabel("share of norm in persona space (%)", fontsize=9.5,
                   color=TEXT_M, labelpad=2)

    # ---- right: AUC of the full direction vs its persona component ----
    style(axR, g)
    axR.grid(axis="x", color=GRID, linewidth=0.8)
    for y, (model, c, share, full, proj, chance) in zip(ys, ROWS):
        ci = CIS[(model, c)]
        # the two bands overlap into a darker core where they agree
        axR.plot([full - ci[0], full + ci[0]], [y, y], color=c, lw=g["band"],
                 alpha=0.18, solid_capstyle="round", zorder=2)
        axR.plot([proj - ci[1], proj + ci[1]], [y, y], color=c, lw=g["band"],
                 alpha=0.30, solid_capstyle="round", zorder=2)
        axR.plot([full, proj], [y, y], color=c, lw=1.6, alpha=0.55, zorder=3)
        axR.scatter([full], [y], s=g["circ"], facecolor="white", edgecolor=c,
                    linewidth=1.5, zorder=4)
        axR.scatter([proj], [y], s=g["star"], marker="*", color=c,
                    edgecolor="white", linewidth=0.6, zorder=5)
    axR.set_xlim(0.70, 0.90)
    axR.set_ylim(*ylim)
    axR.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90])
    axR.set_xlabel("jailbreak detection AUC", fontsize=9.5, color=TEXT_M,
                   labelpad=2)
    if g["notes"]:
        t = axR.text(0.706, ylim[0] + 0.03, "chance = 0.5", fontsize=7.2,
                     color=TEXT_M, ha="left", va="bottom")
        halo(t)
    for dx in (0.0, 0.018):      # axis-break marks: the axis starts at 0.70
        axR.plot([0.012 + dx, 0.03 + dx], [-0.02, 0.02],
                 transform=axR.transAxes, color=SPINE, lw=1.1,
                 clip_on=False, zorder=6)

    fig.tight_layout(pad=0.15)

    # Open the frame until nothing touches it. The padding needed is set by
    # rendered ink -- cap height of the headers at the top, bar edge at the
    # bottom -- so it is measured, not guessed: an eyeballed pad in data units
    # is what put the header caps on the top spine.
    rend = fig.canvas.get_renderer()
    dpi = fig.dpi / 72.0
    lo, hi = ylim
    for _ in range(6):
        fig.canvas.draw()
        need_top = need_bot = 0.0
        for ax in (axL, axR):
            boxes = [b for _n, b in ink(ax, rend)]
            if not boxes:
                continue
            span = (hi - lo) / (ax.bbox.height / dpi)     # data units per pt
            need_top = max(need_top, span * (
                FRAME_PT - (ax.bbox.y1 - max(b.y1 for b in boxes)) / dpi))
            need_bot = max(need_bot, span * (
                FRAME_PT - (min(b.y0 for b in boxes) - ax.bbox.y0) / dpi))
        if need_top <= 0.001 and need_bot <= 0.001:
            break
        hi += max(need_top, 0.0)
        lo -= max(need_bot, 0.0)
        axL.set_ylim(lo, hi)

    # Stacked, frameless legend at the top left of the AUC panel, at the
    # largest size that keeps every one of its rows PAD points clear of the
    # data. Two subtleties:
    #   * handlelength is cut to 0.9 -- the default reserves a full line's
    #     worth of handle column for what is a single marker, which pushed
    #     both labels ~9pt to the right for nothing
    #   * the test is per row, not on the legend's bounding box. The box spans
    #     the width of the longer row, so it overlaps the top row's markers
    #     while no ink does; testing it whole would reject placements that
    #     read perfectly.
    PAD = 2.5
    anchor_x = FRAME_PT / (axR.bbox.width / dpi)
    anchor_y = 1.0 - FRAME_PT / (axR.bbox.height / dpi)
    placed = None
    for fs in [8.5, 8.0, 7.5, 7.0, 6.5, 6.0]:
        handles = [
            Line2D([0], [0], marker="*", ls="", markerfacecolor="#666",
                   markeredgecolor="white", markersize=fs * 1.45,
                   label="persona component"),
            Line2D([0], [0], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="#666", markersize=fs * 0.80,
                   label="full direction"),
        ]
        leg = axR.legend(handles=handles, loc="upper left", ncol=1, fontsize=fs,
                         frameon=False, borderpad=0.0, handletextpad=0.50,
                         labelspacing=0.25, borderaxespad=0.0, handlelength=0.9,
                         bbox_to_anchor=(anchor_x, anchor_y))
        for t in leg.get_texts():
            t.set_color(TEXT_M)
        fig.canvas.draw()
        lb = leg.get_window_extent(rend)
        rows = [Bbox([[lb.x0, tb.y0], [tb.x1, tb.y1]])
                for tb in (t.get_window_extent(rend) for t in leg.get_texts())]
        data = [b for n, b in ink(axR, rend) if n != "legend"]
        if (lb.x0 >= axR.bbox.x0 and lb.x1 <= axR.bbox.x1
                and not any(not (r.x1 + PAD * dpi < b.x0
                                 or r.x0 - PAD * dpi > b.x1
                                 or r.y1 + PAD * dpi < b.y0
                                 or r.y0 - PAD * dpi > b.y1)
                            for r in rows for b in data)):
            placed = fs
            break
        leg.remove()
    return fig, placed


def collisions(fig):
    """Everything the eye catches and a text-vs-text test does not.

    Three separate failures, because each has a different cause:
      * two text boxes touching   -- row pitch too tight for the labels
      * two bars touching         -- row pitch too tight for the bars, which
        happens well before the labels collide (the labels have halos and sit
        over the bars, so they never register a hit)
      * a header sitting on a bar -- header offset too small
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axL = fig.axes[0]
    bad = []

    items = []
    for ax in fig.axes:
        for t in list(ax.texts) + list(ax.get_yticklabels()) \
                + list(ax.get_xticklabels()) + [ax.xaxis.label]:
            if t.get_text().strip():
                items.append((t.get_text().strip()[:24],
                              t.get_window_extent(rend)))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (na, a), (nb, b) = items[i], items[j]
            if not (a.x1 < b.x0 or a.x0 > b.x1 or a.y1 < b.y0 or a.y0 > b.y1):
                bad.append(f"text {na!r} vs {nb!r}")

    dpi = fig.dpi / 72.0
    bars = sorted((p.get_window_extent(rend) for p in axL.patches),
                  key=lambda b: b.y0)
    for a, b in zip(bars, bars[1:]):
        gap = (b.y0 - a.y1) / dpi
        if gap < BAR_GAP_PT:
            bad.append(f"bars {gap:.1f}pt apart (< {BAR_GAP_PT})")
    for t in [t for t in axL.texts if t.get_text() in HEAD_NAMES]:
        tb = t.get_window_extent(rend)
        for bb in bars:
            if tb.x1 > bb.x0 and tb.x0 < bb.x1:
                gap = (tb.y0 - bb.y1) / dpi
                if -tb.height / dpi < gap < HEAD_CLEAR_PT:
                    bad.append(f"header {t.get_text()!r} {gap:.1f}pt over a bar")

    for ax in fig.axes:
        for name, bb in ink(ax, rend):
            edges = [("top", (ax.bbox.y1 - bb.y1) / dpi),
                     ("bottom", (bb.y0 - ax.bbox.y0) / dpi)]
            if name == "legend":
                # only the legend is checked sideways: bars start on the left
                # spine by construction, so a blanket horizontal test would
                # flag the one thing that is meant to touch
                edges += [("left", (bb.x0 - ax.bbox.x0) / dpi),
                          ("right", (ax.bbox.x1 - bb.x1) / dpi)]
            for edge, gap in edges:
                if gap < FRAME_PT - 0.05:
                    bad.append(f"{name!r} {gap:.1f}pt from the {edge} "
                               f"frame (< {FRAME_PT})")
    return bad


def pdf_size(path):
    d = path.read_bytes()
    b = [float(x) for x in
         re.search(rb"/MediaBox \[([\d.\s]+)\]", d).group(1).split()]
    return b[2] - b[0], b[3] - b[1]


def save_at_linewidth(g, out, stem):
    """Render, measure, rescale the canvas so the saved PDF is exactly
    \\linewidth wide. The tight bbox adds a text-driven margin that is roughly
    constant in inches, so a few passes converge."""
    figw = 5.9
    for _ in range(5):
        fig, placed = build(g, figw)
        bad = collisions(fig)
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
        w, h = pdf_size(out / f"{stem}.pdf")
        if abs(w - LINEWIDTH_PT) < 0.6:
            fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight",
                        pad_inches=0.01)
            plt.close(fig)
            return w, h, placed, bad
        figw += (LINEWIDTH_PT - w) / 72.0
        plt.close(fig)
    return w, h, placed, bad


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="roomy", choices=sorted(VARIANTS))
    p.add_argument("--outdir", default=str(OUT))
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    w, h, placed, bad = save_at_linewidth(VARIANTS[a.variant], out,
                                          "fig_direction_story")
    print(f"fig_direction_story [{a.variant}]  {w:.1f} x {h:.1f} pt   "
          f"legend {placed}pt   {'clean' if not bad else bad}")
