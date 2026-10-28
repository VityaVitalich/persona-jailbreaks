#!/usr/bin/env python3
"""
plot_activation_scatter.py

Visualizes the jailbreak activation space at layer 16 as a 2D scatter plot.
Each point is one (behavior, jailbreak template) pair's pre-generation activation,
colored by whether the model was jailbroken or refused.

Overlays trait vector arrows showing where key persona directions point
in the same 2D space.

Produces TWO plots side by side:
  Left:  Jailbreak PCA axes (PC1/PC2 from PCA fit on harmbench activations)
  Right: Fresh PCA axes (PCA fit fresh for visualization, same data)

Usage:
  uv run full_trait_tools/plot_activation_scatter.py

Reproducing Figure 2 of the paper (both panels + the legend they share). The
Gemma invocation was never recorded anywhere and had to be reconstructed from
the caption, which is why it is written out here: layer 30, HarmBench +
WildJailbreak, PCA basis fit on HarmBench alone and unstandardised (the basis
of tab:pca-decomp-gemma), boundary refit in the plotted plane because the
full-dimensional classifier is near-orthogonal to it. Since 2026-09-14 every
arrow is oriented into the jailbroken half-plane and drawn at one length, so
--flip_axis is no longer needed; ticks and PC titles are on by default.

  FT=full_trait_output
  COMMON="--single --no_legend --label_scale 0.78 --figh 2.30"

  python3.13 full_trait_tools/plot_activation_scatter.py $COMMON \
    --output_path $FT/plots/paper_figs/fig_jailbreak_pca.pdf

  python3.13 full_trait_tools/plot_activation_scatter.py $COMMON \
    --classified_path $FT/harmbench_activations_gemma/classified_responses.jsonl \
                      $FT/wildjailbreak_activations_gemma/classified_responses.jsonl \
    --activations_path $FT/harmbench_activations_gemma/activations.pt \
                       $FT/wildjailbreak_activations_gemma/activations.pt \
    --attack_types human_jailbreak wjb_adversarial \
    --pca_fit_attack_types human_jailbreak \
    --layer 30 --vector_row 3 --no_scale --boundary_fit_2d \
    --trait_vectors_dir $FT/gemma_vectors/filter_matched_pairs_ge_50_count_ge_10_total \
    --axis_path $FT/gemma_axes/all_traits_no_filter/assistant_axis_pc1.pt \
    --traits romantic emotional literal stoic dispassionate \
    --output_path $FT/plots/paper_figs/fig_jailbreak_pca_gemma.pdf

  python3.13 full_trait_tools/plot_activation_scatter.py --legend_strip \
    --output_path $FT/plots/paper_figs/fig_jailbreak_pca_legend.pdf

Geometry: each panel sits at 0.485\linewidth = 192.5pt, so the 4.8in drawing
width lands at scale 0.5688. The legend strip is drawn at the panels' own font
size and MUST be included at that same scale (not at a width), or its text
comes out a different size from theirs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LAYER = 16
RANDOM_SEED = 42

# Traits to show as arrows
TRAITS_TO_SHOW = [
    "naive",
    "essentialist",
    "utilitarian",
    "absolutist",
    "progressive",
]
AXIS_NAME = "assistant_axis"           # key of the PC1-of-trait-space axis file
AXIS_LABEL = "Principal Persona Axis"  # how the paper names it (renamed 2026-09-14)

# ── Palette ("journal": plain white, Okabe-Ito-ish warm/cool pair) ────────────
FIG_BG    = "#FFFFFF"
PANEL_BG  = "#FFFFFF"
C_REFUSED = "#7FB3D8"   # sky blue  (refused prompts)
C_JAILBRK = "#E8917D"   # soft coral (jailbroken prompts)
C_DEGEN   = "#BFBAB0"   # muted grey
# Arrows are directions, not prompts, so they take hues the points never use:
# near-black for trait directions, dark green for the principal persona axis.
# (The old brick-red / navy arrows sat next to the coral / sky-blue points and
# read as a third and fourth kind of point.)
ARROW_DARK = "#B8860B"  # trait-direction arrows (dark gold; variant C, 2026-09-14)
AXIS_GREEN = "#0F766E"  # the principal persona axis (teal)
ARROW_LEN_FRAC = 0.60   # common arrow length, as a fraction of panel height
TEXT_DARK = "#222222"
TEXT_MUT  = "#555555"
GRID      = "#E7E7E7"
SPINE     = "#C9C9C9"
BOUNDARY  = "#8A8A8A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": FIG_BG,
    "savefig.facecolor": FIG_BG,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trait_vector(vectors_dir: Path, trait: str, layer: int) -> np.ndarray:
    path = vectors_dir / f"{trait}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"][layer].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def load_axis(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    v = data["axis"][layer].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def load_hyperplane(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


# ── Plot one scatter panel ─────────────────────────────────────────────────────

def _panel_limits(X_2d: np.ndarray):
    """Robust limits from the 1st/99th percentile so outliers don't shrink the
    cluster; keeps the bulk of the data big and visible."""
    lo = np.percentile(X_2d, 1, axis=0)
    hi = np.percentile(X_2d, 99, axis=0)
    pad = 0.05 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def titlecase(text):
    """Every Word Capitalised, as the paper's other figures label things."""
    return " ".join(w[:1].upper() + w[1:] for w in text.split())


def _seg_hits_rect(a, b, r, pad):
    """Does segment a-b come within `pad` of rectangle r? Liang-Barsky on the
    rect grown by pad -- a wide label has to be tested as a rectangle, not as a
    disc around its centre, or an arrow crossing one end goes unnoticed."""
    x0, y0, x1, y1 = r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p_, q_ in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                   (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p_ == 0:
            if q_ < 0:
                return False
            continue
        t = q_ / p_
        if p_ < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _place_labels(ax, plabels, arrows_xy, pad_pt=2.0, arrow_pt=3.0):
    """Choose each label's position by search rather than by nudging.

    The original pass separated labels from labels only, and measured the gap
    in data units, so the visual spacing shrank with the panel and an arrow
    could run straight through a word. Here every label keeps its own arrow's
    direction but is tried at a few distances out along it and a few
    perpendicular offsets; the first candidate that clears every arrow and
    every already-placed label wins. Text boxes do not change size when moved,
    so each is measured once and the rectangle is translated -- the whole
    search costs one draw.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dpi = fig.dpi / 72.0
    T = ax.transData.transform
    segs = [(T(a), T(b)) for a, b in arrows_xy]
    frame = ax.get_window_extent()

    o = T((0.0, 0.0))
    for pl in plabels:
        bb = pl["artist"].get_window_extent(rend)
        pl["w"], pl["h"] = bb.width, bb.height
        pl["anchor"] = T((pl["x"], pl["y"]))
        # the arrow direction has to be re-derived in display space: the panel
        # is far from square, so a unit vector in data units points somewhere
        # else on the page
        v = T(pl["dir"]) - o
        n = float(np.hypot(*v)) or 1.0
        pl["ddir"] = (float(v[0] / n), float(v[1] / n))

    def rect(pl, px, py):
        w, h = pl["w"], pl["h"]
        x0 = px if pl["ha"] == "left" else px - w
        return (x0, py - h / 2, x0 + w, py + h / 2)

    def inside(r):
        return (r[0] >= frame.x0 + 1 and r[2] <= frame.x1 - 1
                and r[1] >= frame.y0 + 1 and r[3] <= frame.y1 - 1)

    placed = []
    # longest first: the wide labels have the fewest legal homes
    for pl in sorted(plabels, key=lambda p_: -p_["w"]):
        d = pl["ddir"]
        best, best_cost, best_ha = None, None, pl["ha"]
        for ha in (pl["ha"], "left" if pl["ha"] == "right" else "right"):
          pl["ha"] = ha
          for along in (0, 5, 10, 15, 20, 26, 32, 40):
            for perp in (0, -7, 7, -14, 14, -21, 21, -28, 28,
                         -36, 36, -44, 44):
                px = pl["anchor"][0] + (d[0] * along - d[1] * perp) * dpi
                py = pl["anchor"][1] + (d[1] * along + d[0] * perp) * dpi
                r = rect(pl, px, py)
                cost = 0
                if not inside(r):
                    cost += 1000
                for sa, sb in segs:
                    if _seg_hits_rect(sa, sb, r, arrow_pt * dpi):
                        cost += 100
                for q in placed:
                    if _seg_hits_rect((q[0], q[1]), (q[2], q[1]), r, pad_pt * dpi) \
                            or _seg_hits_rect((q[0], q[3]), (q[2], q[3]), r,
                                              pad_pt * dpi) \
                            or (r[0] < q[2] and q[0] < r[2]
                                and r[1] < q[3] and q[1] < r[3]):
                        cost += 100
                cost += 0.5 * (abs(along) + abs(perp))   # prefer staying put
                if best_cost is None or cost < best_cost:
                    best, best_cost, best_ha = (px, py), cost, ha
                if cost < 1:
                    break
            if best_cost is not None and best_cost < 1:
                break
          if best_cost is not None and best_cost < 1:
              break
        pl["ha"] = best_ha
        pl["artist"].set_ha(best_ha)
        pl["artist"].set_position(ax.transData.inverted().transform(best))
        pl["rect"] = rect(pl, best[0], best[1])
        placed.append(pl["rect"])
        pl["cost"] = best_cost

    bad = [pl["name"] for pl in plabels if pl["cost"] >= 100]
    if bad:
        print("  LABEL COLLISIONS: " + ", ".join(bad))
    return bad


def plot_panel(
    ax,
    X_2d: np.ndarray,
    labels: np.ndarray,      # 0=refused, 1=jailbroken, 2=degenerate
    trait_arrows: Dict[str, np.ndarray],  # name -> 2D projected vector
    w_2d: np.ndarray,
    title: str = None,
    b_2d: float = 0.0,
    legend_ncol: int = 1,
    legend: bool = True,
    label_scale: float = 1.0,
    xlabel: str = None,
    xtitle: str = None,
    ytitle: str = None,
    show_ticks: bool = False,
    orient_w: np.ndarray = None,
) -> None:

    ax.set_facecolor(PANEL_BG)
    xlim, ylim = _panel_limits(X_2d)

    # 2. grid + frame
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=1)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    if show_ticks:
        # Ticks in PCA units: they show the cloud is mean-centred (the origin,
        # where every arrow starts, is the data mean) and give the axes a scale.
        # the paper prints the panel at ~0.57 scale, so 10.5pt here is ~6pt
        ax.tick_params(length=2.5, width=0.7, color=SPINE, labelcolor=TEXT_MUT,
                       labelsize=10.5, pad=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        if xtitle:
            ax.set_xlabel(xtitle, fontsize=12, color=TEXT_DARK, labelpad=2)
        if ytitle:
            ax.set_ylabel(ytitle, fontsize=12, color=TEXT_DARK, labelpad=2)
    else:
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

    # 3. scatter
    m_ref = labels == 0
    m_jb  = labels == 1
    m_deg = labels == 2
    if m_deg.any():
        ax.scatter(X_2d[m_deg, 0], X_2d[m_deg, 1], c=C_DEGEN, s=10,
                   alpha=0.35, linewidths=0, zorder=2)
    ax.scatter(X_2d[m_ref, 0], X_2d[m_ref, 1], c=C_REFUSED, s=18, alpha=0.55,
               edgecolors="white", linewidths=0.3, zorder=3)
    ax.scatter(X_2d[m_jb, 0], X_2d[m_jb, 1], c=C_JAILBRK, s=18, alpha=0.55,
               edgecolors="white", linewidths=0.3, zorder=3)

    # 4. classifier boundary (w·x + b = 0  ->  y = -(w0 x + b)/w1)
    if abs(w_2d[1]) > 1e-6:
        xr = np.array(xlim)
        ax.plot(xr, -(w_2d[0] * xr + b_2d) / w_2d[1], ls=(0, (6, 4)),
                color=BOUNDARY, lw=1.4, alpha=0.75, zorder=4)

    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])

    arrows_xy = []

    def text_width(text):
        # 13pt bold DejaVu ≈ 0.11in/char; convert to data units via axes size
        ax_w_in = ax.figure.get_size_inches()[0] * ax.get_position().width
        return 0.115 * len(text) * span / ax_w_in

    def label_ha(lx, text, dx):
        # extend outward from the origin, but flip inward near a panel edge
        tw = text_width(text)
        if dx >= 0:
            return "right" if lx + tw > xlim[1] else "left"
        return "left" if lx - tw < xlim[0] else "right"

    def clamp_lx(lx, text, ha):
        # keep the whole label inside the panel
        tw = text_width(text)
        if ha == "right":
            return max(lx, xlim[0] + tw + 0.01 * span)
        return min(lx, xlim[1] - tw - 0.01 * span)

    def draw_label(lx, ly, text, color, fontsize, ha, va):
        t = ax.text(lx, ly, text, color=color, fontsize=fontsize,
                    fontweight="bold", ha=ha, va=va, zorder=8, clip_on=True)
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])
        return t

    # 5. arrows. Every direction is a unit vector whose sign is arbitrary (a
    #    trait vector is pos-minus-neg by convention, PC1 carries the solver's
    #    sign) and whose in-plane length only says how much of it lies in the
    #    plotted plane. Neither is what the figure is about, so each arrow is
    #    oriented into the jailbroken half-plane of the classifier and drawn at
    #    one common display length; only its angle carries information.
    ax_w_in = ax.figure.get_size_inches()[0] * ax.get_position().width
    ax_h_in = ax.figure.get_size_inches()[1] * ax.get_position().height
    sx = ax_w_in / (xlim[1] - xlim[0])      # inches per data unit, x
    sy = ax_h_in / (ylim[1] - ylim[0])      # inches per data unit, y
    L_in = ARROW_LEN_FRAC * ax_h_in

    def draw_arrow(vec, color):
        v = np.asarray(vec, dtype=float)
        if orient_w is not None and float(np.dot(v, orient_w)) < 0:
            v = -v
        disp = np.array([v[0] * sx, v[1] * sy])
        disp /= (np.linalg.norm(disp) + 1e-12)
        tip = np.array([disp[0] * L_in / sx, disp[1] * L_in / sy])
        d = tip / (np.linalg.norm(tip) + 1e-12)
        arrows_xy.append(((0.0, 0.0), (float(tip[0]), float(tip[1]))))
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(0, 0), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.4, mutation_scale=18,
                                    shrinkA=0, shrinkB=0))
        return d, tip

    def add_label(d, tip, text, color, fs):
        lx = tip[0] + d[0] * 0.03 * span
        ly = tip[1] + d[1] * 0.03 * span
        ha = label_ha(lx, text, d[0])
        plabels.append({"x": clamp_lx(lx, text, ha), "y": ly, "name": text,
                        "ha": ha, "color": color, "fs": fs,
                        "dir": (float(d[0]), float(d[1])),
                        "tip": (float(tip[0]), float(tip[1]))})

    plabels = []
    persona = [(n, v) for n, v in trait_arrows.items() if n != AXIS_NAME]
    persona.sort(key=lambda nv: np.arctan2(nv[1][1], nv[1][0]))
    for name, vec_2d in persona:
        d, tip = draw_arrow(vec_2d, ARROW_DARK)
        add_label(d, tip, titlecase(name), TEXT_DARK, 13 * label_scale)
    if AXIS_NAME in trait_arrows:
        d, tip = draw_arrow(trait_arrows[AXIS_NAME], AXIS_GREEN)
        add_label(d, tip, AXIS_LABEL, AXIS_GREEN, 13.5 * label_scale)

    for pl in plabels:
        pl["artist"] = draw_label(pl["x"], pl["y"], pl["name"], pl["color"],
                                  pl["fs"], pl["ha"], "center")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12.5, fontweight="bold",
                      color=TEXT_DARK, labelpad=3)
    _place_labels(ax, plabels, arrows_xy)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_DARK,
                     pad=10)

    handles = [
        Line2D([0], [0], marker="o", ls="", ms=9, mec="white", mew=0.4,
               color=C_JAILBRK, label="Jailbroken"),
        Line2D([0], [0], marker="o", ls="", ms=9, mec="white", mew=0.4,
               color=C_REFUSED, label="Refused"),
        Line2D([0], [0], color=BOUNDARY, ls=(0, (6, 4)), lw=1.4,
               label="Classifier Boundary"),
        Line2D([0], [0], color=ARROW_DARK, lw=2.4, marker=">", ms=6,
               markevery=[1], label="Trait Direction"),
        Line2D([0], [0], color=AXIS_GREEN, lw=2.4, marker=">", ms=6,
               markevery=[1], label="Principal Persona Axis"),
    ]
    if not legend:
        return
    # One row instead of three when the panel is short: the stacked box eats
    # the whole upper-right quadrant, which is what stops the panel from
    # getting shorter before the scatter itself does.
    leg = ax.legend(handles=handles, fontsize=11.5,
                    loc="upper center" if legend_ncol > 1 else "upper right",
                    ncol=legend_ncol, columnspacing=1.1, handletextpad=0.4,
                    borderpad=0.35, framealpha=0.9, edgecolor=SPINE)
    leg.get_frame().set_facecolor("white")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified_path", type=str, nargs="+",
        default=["full_trait_output/harmbench_activations/classified_responses.jsonl"])
    parser.add_argument("--activations_path", type=str, nargs="+",
        default=["full_trait_output/harmbench_activations/activations.pt"])
    parser.add_argument("--attack_types", type=str, nargs="+",
        default=["human_jailbreak"])
    parser.add_argument("--hyperplane_path", type=str,
        default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--trait_vectors_dir", type=str,
        default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--axis_path", type=str,
        default="full_trait_output/traits40_axes/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt")
    parser.add_argument("--traits", type=str, nargs="+", default=TRAITS_TO_SHOW)
    parser.add_argument("--output_path", type=str,
        default="full_trait_output/plots/activation_scatter_layer16.png")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--vector_row", type=int, default=None,
        help="Row index into the stored [n_layers, hidden] trait tensors; "
             "defaults to --layer (Llama stores all 32 layers). Gemma tensors "
             "store selected layers only, e.g. layer 30 = row 3.")
    parser.add_argument("--flip_axis", action="store_true",
        help="Negate the assistant axis for display. Its PC1 sign is "
             "solver-arbitrary; use this to keep the display convention "
             "(axis pointing toward refusals) across models.")
    parser.add_argument("--no_scale", action="store_true",
        help="PCA on raw activations without standardization (the basis used "
             "by pca_direction_decomposition.py / the Gemma PCA tables).")
    parser.add_argument("--pca_fit_attack_types", type=str, nargs="+",
        default=None,
        help="Fit the PCA basis only on rows with these attack types (e.g. "
             "pure HarmBench), then project all plotted points into it.")
    parser.add_argument("--boundary_fit_2d", action="store_true",
        help="Draw the boundary of a balanced logistic regression fit on the "
             "plotted 2D coordinates instead of projecting the full-dim "
             "hyperplane. Use when the hyperplane is near-orthogonal to the "
             "plotted plane (e.g. Gemma raw basis).")
    parser.add_argument("--figh", type=float, default=3.85,
        help="Panel height in inches at the 4.8in drawing width. The paper scales the panel to 0.485\\linewidth, so this sets the aspect and therefore how tall the figure block is on the page; everything else keeps its current proportions.")
    parser.add_argument("--legend_ncol", type=int, default=1,
        help="3 lays the legend out as a single row, which frees the upper-right quadrant for a shorter panel.")
    parser.add_argument("--no_legend", action="store_true",
        help="Omit the per-panel legend. Use with --legend_strip when two "
             "panels share one legend in the paper figure.")
    parser.add_argument("--label_scale", type=float, default=1.0,
        help="Multiplies the persona/axis label sizes.")
    parser.add_argument("--xlabel", type=str, default=None,
        help="Text under the panel, e.g. the model it shows.")
    parser.add_argument("--legend_strip", action="store_true",
        help="Render ONLY the legend, as a one-row strip, for use as a "
             "shared legend under both panels. Include it in LaTeX at the "
             "same scale as the panels (see the docstring) so its text "
             "matches theirs.")
    parser.add_argument("--trait_color", type=str, default=None,
        help="Override the trait-arrow colour (hex).")
    parser.add_argument("--axis_color", type=str, default=None,
        help="Override the principal-persona-axis arrow colour (hex).")
    parser.add_argument("--no_ticks", action="store_true",
        help="Hide tick labels and the PC axis titles (the pre-2026-09-14 look).")
    parser.add_argument("--legend_cols", type=int, default=5,
        help="Columns of the --legend_strip.")
    parser.add_argument("--single", action="store_true",
        help="Render only the Jailbreak-PCA panel (single-panel paper figure).")
    args = parser.parse_args()
    global ARROW_DARK, AXIS_GREEN
    if args.trait_color:
        ARROW_DARK = args.trait_color
    if args.axis_color:
        AXIS_GREEN = args.axis_color

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.legend_strip:
        # Drawn at the panels' own font size so that including it at the same
        # scale as a panel puts its text at exactly the panel's text size.
        fig = plt.figure(figsize=(9.6, 0.4), dpi=200)
        handles = [
            Line2D([0], [0], marker="o", ls="", ms=9, mec="white", mew=0.4,
                   color=C_JAILBRK, label="Jailbroken"),
            Line2D([0], [0], marker="o", ls="", ms=9, mec="white", mew=0.4,
                   color=C_REFUSED, label="Refused"),
            Line2D([0], [0], color=BOUNDARY, ls=(0, (6, 4)), lw=1.4,
                   label="Classifier Boundary"),
            Line2D([0], [0], color=ARROW_DARK, lw=2.4, marker=">", ms=6,
                   markevery=[1], label="Trait Direction"),
            Line2D([0], [0], color=AXIS_GREEN, lw=2.4, marker=">", ms=6,
                   markevery=[1], label="Principal Persona Axis"),
        ]
        leg = fig.legend(handles=handles, fontsize=11.5, ncol=args.legend_cols,
                         loc="center", columnspacing=1.3, handletextpad=0.5,
                         handlelength=1.8, borderpad=0.35, framealpha=0.9,
                         edgecolor=SPINE)
        leg.get_frame().set_facecolor("white")
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
        print(f"\nSaved legend strip to {output_path}")
        return

    # ── Load data (one or more classified/activations source pairs) ───────────
    assert len(args.classified_path) == len(args.activations_path)
    layer_key = str(args.layer)
    X_list, labels_list, atype_list = [], [], []
    for cpath, apath in zip(args.classified_path, args.activations_path):
        rows = load_jsonl(Path(cpath))
        rows = [r for r in rows if r.get("attack_type") in args.attack_types]
        print(f"  {cpath}: {len(rows)} pairs kept")
        activations = torch.load(Path(apath), map_location="cpu",
                                 weights_only=False)
        for row in rows:
            pid = row["pair_id"]
            if pid not in activations or layer_key not in activations[pid]:
                continue
            X_list.append(activations[pid][layer_key].float().numpy())
            atype_list.append(row.get("attack_type"))
            jb = row.get("jailbroken")
            if jb is True:
                labels_list.append(1)
            elif jb is False:
                labels_list.append(0)
            else:
                labels_list.append(2)  # unknown/degenerate

    X_all    = np.stack(X_list)
    labels   = np.array(labels_list)
    print(f"  Matrix shape: {X_all.shape}")
    print(f"  Jailbroken: {(labels==1).sum()}, Refused: {(labels==0).sum()}, Other: {(labels==2).sum()}")

    # ── Standardize (unless raw basis requested) ───────────────────────────────
    fit_mask = np.ones(len(X_all), dtype=bool)
    if args.pca_fit_attack_types:
        fit_mask = np.isin(np.array(atype_list), args.pca_fit_attack_types)
        print(f"PCA basis fit on {fit_mask.sum()}/{len(X_all)} rows "
              f"({args.pca_fit_attack_types})")
    if args.no_scale:
        X_scaled = X_all
    else:
        print("Standardizing...")
        scaler = StandardScaler().fit(X_all[fit_mask])
        X_scaled = scaler.transform(X_all)

    # ── Fit jailbreak PCA (4 components, use first 2 for plotting) ─────────────
    print("Fitting jailbreak PCA (n=4)...")
    pca_jb = PCA(n_components=4, random_state=RANDOM_SEED)
    pca_jb.fit(X_scaled[fit_mask])
    X_jb_4d = pca_jb.transform(X_scaled)
    X_jb_2d = X_jb_4d[:, :2]
    print(f"  Var explained: {[f'{100*v:.1f}%' for v in pca_jb.explained_variance_ratio_]}")

    # ── Fit fresh PCA (2 components for visualization) ────────────────────────
    print("Fitting fresh PCA (n=2)...")
    pca_fresh = PCA(n_components=2, random_state=RANDOM_SEED)
    X_fresh_2d = pca_fresh.fit_transform(X_scaled)
    print(f"  Var explained: {[f'{100*v:.1f}%' for v in pca_fresh.explained_variance_ratio_]}")

    # ── Load trait vectors + project to 2D ────────────────────────────────────
    print("Loading trait vectors...")
    vectors_dir = Path(args.trait_vectors_dir)
    row_idx = args.vector_row if args.vector_row is not None else args.layer

    raw_vecs = {}
    for trait in args.traits:
        try:
            raw_vecs[trait] = load_trait_vector(vectors_dir, trait, row_idx)
            print(f"  Loaded: {trait}")
        except FileNotFoundError:
            print(f"  WARNING: {trait} not found")

    try:
        raw_vecs[AXIS_NAME] = load_axis(Path(args.axis_path), row_idx)
        if args.flip_axis:
            raw_vecs[AXIS_NAME] = -raw_vecs[AXIS_NAME]
        print(f"  Loaded: {AXIS_NAME}")
    except FileNotFoundError:
        print(f"  WARNING: {AXIS_NAME} not found")

    # Project each trait vector into PCA space
    # For a unit vector v in original space, its projection onto PC_i is:
    # v_projected[i] = pca.components_[i] · v  (since scaler mean ~ 0 for unit vecs)
    def project_to_pca_2d(vec, pca):
        return np.array([
            np.dot(pca.components_[0], vec),
            np.dot(pca.components_[1], vec),
        ])

    trait_arrows_jb = {
        name: project_to_pca_2d(vec, pca_jb)
        for name, vec in raw_vecs.items()
    }
    trait_arrows_fresh = {
        name: project_to_pca_2d(vec, pca_fresh)
        for name, vec in raw_vecs.items()
    }

    # ── Boundary: projected hyperplane, or logreg fit in the plotted plane ────
    b_jb_2d = 0.0
    if args.boundary_fit_2d:
        from sklearn.linear_model import LogisticRegression
        m01 = labels < 2
        clf2 = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf2.fit(X_jb_2d[m01], labels[m01])
        w_jb_2d = clf2.coef_[0]
        b_jb_2d = float(clf2.intercept_[0])
        w_fresh_2d = w_jb_2d  # only used by the two-panel dev mode
        print("Boundary: balanced logreg fit on the plotted 2D coordinates")
    else:
        print("Loading hyperplane...")
        w_vec = load_hyperplane(Path(args.hyperplane_path))
        w_jb_2d    = project_to_pca_2d(w_vec, pca_jb)
        w_fresh_2d = project_to_pca_2d(w_vec, pca_fresh)

    # Boundary normal pointed at the jailbroken side of the plotted plane, so
    # plot_panel can flip every arrow into that half-plane.
    m_jb, m_ref = labels == 1, labels == 0
    orient_w = np.asarray(w_jb_2d, dtype=float)
    if float(np.dot(orient_w, X_jb_2d[m_jb].mean(0) - X_jb_2d[m_ref].mean(0))) < 0:
        orient_w = -orient_w
    evr = pca_jb.explained_variance_ratio_
    xtitle = f"PC1 ({100 * evr[0]:.1f}%)"
    ytitle = f"PC2 ({100 * evr[1]:.1f}%)"

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("Plotting...")
    if args.single:
        # No in-figure title / axis labels: the paper caption carries those.
        # Sized for a half-linewidth panel (paired with the trait-AUC dot plot).
        fig, ax = plt.subplots(figsize=(4.8, args.figh), dpi=200)
        plot_panel(
            ax, X_jb_2d, labels,
            trait_arrows_jb, w_jb_2d,
            b_2d=b_jb_2d, legend_ncol=args.legend_ncol,
            legend=not args.no_legend, label_scale=args.label_scale,
            xlabel=args.xlabel, xtitle=xtitle, ytitle=ytitle,
            show_ticks=not args.no_ticks, orient_w=orient_w,
        )
        fig.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved to {output_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=200)
    fig.suptitle(
        f"Pre-generation activation space  ·  Layer {args.layer}",
        fontsize=15, fontweight="bold", color=TEXT_DARK, y=1.02,
    )

    plot_panel(
        axes[0], X_jb_2d, labels,
        trait_arrows_jb, w_jb_2d,
        title="Jailbreak PCA",
    )

    plot_panel(
        axes[1], X_fresh_2d, labels,
        trait_arrows_fresh, w_fresh_2d,
        title="Fresh PCA",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
