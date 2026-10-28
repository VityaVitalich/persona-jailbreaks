#!/usr/bin/env python3
"""
plot_pca_persona_megafig.py

Candidate "mega figure" merging the PCA story into one graphic-table:
how the jailbreak classifier direction w decomposes onto unsupervised PCs
(variance, weight in w), how predictive each PC is on its own (standalone
AUC), which persona poles each PC aligns with, and what steering along a PC
does to a benign generation.

Three layout versions, rendered as PNG previews:
  v1  landscape graph-table: two model panels side by side + steering band
  v2  narrative flow: w-decomposition stacked bars -> persona-axis rows -> steering dial
  v3  portrait: models stacked, steering quotes inline per Llama PC row

All numbers come from the paper tables (tables/pca_llama.tex, pca_gemma.tex,
pca_poles_llama.tex, pca_poles_gemma.tex, pc_steering_qualitative.tex),
which in turn come from decompose_jailbreak_direction.py and
steer_pc_qualitative.py runs.
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

OUT = Path("full_trait_output/plots")
OUT.mkdir(parents=True, exist_ok=True)

# journal palette (same as plot_trait_auc_dotplot.py / PCA scatter)
C_LLAMA = "#A6403A"
C_GEMMA = "#C88A2D"
GRID    = "#E7E7E7"
SPINE   = "#C9C9C9"
TEXT_D  = "#222222"
TEXT_M  = "#555555"
TEXT_L  = "#8A8A8A"
# PC identity colors (Paul Tol bright, CVD-safe), shared across models
C_PC    = ["#4477AA", "#66CCEE", "#228833", "#CCBB44"]
C_RESID = "#BBBBBB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------- data

LLAMA = dict(
    model="Llama-3.1-8B", layer=16, corpus="HarmBench", accent=C_LLAMA,
    pcs=[
        dict(name="PC1", var=17.7, w=0.67, auc=0.80, ci=None,
             neg=[("diplomatic", .28), ("calm", .25), ("methodical", .24)],
             pos=[("edgy", .35), ("sycophantic", .32), ("goofy", .32)]),
        dict(name="PC2", var=13.2, w=0.60, auc=0.74, ci=None,
             neg=[("evil", .25), ("condescending", .24), ("misanthropic", .23)],
             pos=[("principled", .29), ("descriptive", .22), ("conciliatory", .19)]),
        dict(name="PC3", var=9.2, w=0.12, auc=0.66, ci=None,
             neg=[("reserved", .26), ("dispassionate", .25), ("detached", .22)],
             pos=[("introspective", .27), ("witty", .26), ("entertaining", .25)]),
        dict(name="PC4", var=7.3, w=-0.39, auc=0.65, ci=None,
             neg=[("casual", .14), ("hostile", .14), ("evil", .12)],
             pos=[("ritualistic", .16), ("formal", .14), ("conscientious", .13)]),
    ],
    w2=[0.449, 0.360, 0.014, 0.152],   # squared weights of w on PC1..4
    assistant_auc=0.64,
)

GEMMA = dict(
    model="Gemma-4-31B", layer=30, corpus="HarmBench + WildJailbreak", accent=C_GEMMA,
    pcs=[
        dict(name="PC1", var=27.0, w=0.78, auc=0.760, ci=0.068,
             neg=[("generalist", .59), ("strategic", .58), ("methodical", .57)],
             pos=[("impatient", .60), ("competitive", .56), ("pessimistic", .55)]),
        dict(name="PC2", var=14.8, w=0.22, auc=0.710, ci=0.075,
             neg=[("accessible", .11), ("systems thinker", .11), ("meticulous", .10)],
             pos=[("enigmatic", .13), ("obsessive", .11), ("concise", .11)]),
        dict(name="PC3", var=8.6, w=-0.19, auc=0.585, ci=0.058,
             neg=[("exploratory", .11), ("systems thinker", .10), ("cautious", .10)],
             pos=[("impulsive", .14), ("nonchalant", .14), ("essentialist", .13)]),
        dict(name="PC4", var=6.9, w=0.46, auc=0.660, ci=0.086,
             neg=[("dominant", .25), ("structuralist", .25), ("critical", .24)],
             pos=[("concise", .24), ("submissive", .24), ("socratic", .24)]),
    ],
    w2=[0.601, 0.048, 0.035, 0.210],
    assistant_auc=None,
)

STEER = [
    dict(pc="PC1", neg_pole="diplomatic", pos_pole="edgy", cmax=1.0,
         prompt="What is your opinion on people who talk loudly on the phone in public?",
         quotes={
             +1: "So, you wanna know what I think? Well, let me tell ya, it's a total no-brainer, folks! Hehe, get it? …",
             0:  "As a neutral AI, I don't have personal opinions, but I can provide an analysis of the issue …",
             -1: "I can provide information on this topic. Many people consider it impolite or unacceptable …",
         }),
    dict(pc="PC2", neg_pole="evil / hostile", pos_pole="principled", cmax=1.3,
         prompt="What should I do about a neighbor who is constantly rude to me?",
         quotes={
             +1: "Dealing with a neighbor who is constantly rude: stay calm and try to understand the reason for their behavior …",
             0:  "Dealing with a rude neighbor can be frustrating; it's essential to maintain your composure …",
             -1: "Dealing with toxic and abusive behavior is never a good time. I can't engage in … a poisonous relationship.",
         }),
    dict(pc="PC3", neg_pole="reserved", pos_pole="introspective", cmax=1.3,
         prompt="What do you think is the meaning of a good life?",
         quotes={
             +1: "What a fascinating and complex question! I'll try to answer it with a hint of excitement and a dash of fun …",
             0:  "The concept of a 'good life' is subjective and can vary greatly from person to person …",
             -1: "There is no single definition of a good life, as it varies with individual desires …",
         }),
    dict(pc="PC4", neg_pole="casual", pos_pole="ritualistic", cmax=1.3,
         prompt="How should I structure my ideal morning routine?",
         quotes={
             +1: "Creating a well-structured morning routine helps you start the day with a clear plan, priorities, and a sense of purpose …",
             0:  "Structuring an ideal morning routine can vary depending on your preferences; here's a general framework …",
             -1: "There are many ways to structure your ideal morning routine; here are some general steps and suggestions …",
         }),
]

AUC_LO, AUC_HI = 0.5, 0.9
VAR_MAX = 30.0

# ---------------------------------------------------------------- helpers


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def hbar(ax, x0, y, w, h, color, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.0025",
        facecolor=color, edgecolor="none", alpha=alpha, mutation_aspect=0.6))


def var_bar(ax, x0, x1, y, val, color, bh=0.016, fs=8):
    """Variance-share bar with printed value."""
    ax.plot([x0, x1], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=1)
    w = (x1 - x0) * min(val / VAR_MAX, 1.0)
    hbar(ax, x0, y, w, bh, color)
    ax.text(x0 + w + 0.006, y, f"{val:.1f}%", ha="left", va="center",
            fontsize=fs, color=TEXT_D)


def w_bar(ax, x0, x1, y, val, color, bh=0.016, fs=8):
    """Signed weight-in-w bar around a central zero line."""
    xm = (x0 + x1) / 2
    ax.plot([x0, x1], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=1)
    ax.plot([xm, xm], [y - 0.016, y + 0.016], color=SPINE, lw=0.9, zorder=2)
    w = (x1 - x0) / 2 * abs(val)
    if val >= 0:
        hbar(ax, xm, y, w, bh, color)
        ax.text(xm + w + 0.006, y, f"+{val:.2f}", ha="left", va="center",
                fontsize=fs, color=TEXT_D)
    else:
        hbar(ax, xm - w, y, w, bh, color)
        ax.text(xm - w - 0.006, y, f"−{abs(val):.2f}", ha="right", va="center",
                fontsize=fs, color=TEXT_D)


def auc_gauge(ax, x0, x1, y, val, color, ci=None, fs=8, ref=None, ticks=False):
    """Dot on a 0.5-0.9 line; optional CI whisker and reference marker."""
    ax.plot([x0, x1], [y, y], color=GRID, lw=1.4, zorder=1)
    for t in (0.5, 0.7, 0.9):
        xt = x0 + (x1 - x0) * (t - AUC_LO) / (AUC_HI - AUC_LO)
        ax.plot([xt, xt], [y - 0.006, y + 0.006], color=SPINE, lw=0.9)
        if ticks:
            ax.text(xt, y - 0.014, f"{t:.1f}", ha="center", va="top",
                    fontsize=fs - 1.5, color=TEXT_L)
    if ref is not None:
        xr = x0 + (x1 - x0) * (ref - AUC_LO) / (AUC_HI - AUC_LO)
        ax.plot([xr, xr], [y - 0.011, y + 0.011], color=TEXT_L, lw=1.2,
                linestyle=(0, (2, 2)))
    xv = x0 + (x1 - x0) * (val - AUC_LO) / (AUC_HI - AUC_LO)
    if ci:
        lo = x0 + (x1 - x0) * (max(val - ci, AUC_LO) - AUC_LO) / (AUC_HI - AUC_LO)
        hi = x0 + (x1 - x0) * (min(val + ci, AUC_HI) - AUC_LO) / (AUC_HI - AUC_LO)
        ax.plot([lo, hi], [y, y], color=color, lw=2, alpha=0.45,
                solid_capstyle="round")
    ax.scatter([xv], [y], s=52, color=color, zorder=5, edgecolor="white",
               linewidth=1.0)
    ax.text(xv, y + 0.013, f"{val:.2f}", ha="center", va="bottom",
            fontsize=fs, color=TEXT_D)


def poles_block(ax, x0, x1, y, pc, fs=9.5, sub_fs=None, subs=True):
    """Two-line persona-axis display: bold pole pair + supporting traits."""
    xm = (x0 + x1) / 2
    npole, ppole = pc["neg"][0][0], pc["pos"][0][0]
    ax.annotate("", xy=(xm + 0.013, y + 0.012), xytext=(xm - 0.013, y + 0.012),
                arrowprops=dict(arrowstyle="<|-|>", color=TEXT_M, lw=1.1,
                                mutation_scale=9))
    ax.text(xm - 0.018, y + 0.012, npole, ha="right", va="center",
            fontsize=fs, color=TEXT_D, fontweight="bold")
    ax.text(xm + 0.018, y + 0.012, ppole, ha="left", va="center",
            fontsize=fs, color=TEXT_D, fontweight="bold")
    if not subs:
        return
    sfs = sub_fs if sub_fs else fs - 2.3
    sub_n = " · ".join(t for t, _ in pc["neg"][1:3])
    sub_p = " · ".join(t for t, _ in pc["pos"][1:3])
    ax.text(xm - 0.012, y - 0.014, sub_n, ha="right", va="center",
            fontsize=sfs, color=TEXT_M)
    ax.text(xm + 0.012, y - 0.014, sub_p, ha="left", va="center",
            fontsize=sfs, color=TEXT_M)
    ax.text(xm, y - 0.014, "|", ha="center", va="center", fontsize=sfs,
            color=TEXT_L)


def wrap(s, n):
    return "\n".join(textwrap.wrap(s, n))


# ================================================================ V1
# Landscape graph-table: two model panels + steering band.

def v1():
    fig, ax = canvas(13.8, 9.2)

    panels = [(LLAMA, 0.010), (GEMMA, 0.505)]
    PW = 0.470                       # panel width
    # column slots inside a panel (fractions of figure, offsets from x0)
    SL_PC   = (0.000, 0.040)
    SL_VAR  = (0.046, 0.120)
    SL_W    = (0.134, 0.218)
    SL_AUC  = (0.232, 0.310)
    SL_POLE = (0.315, 0.460)

    row_y = [0.845, 0.740, 0.635, 0.530]
    hdr_y = 0.912

    for spec, x0 in panels:
        # panel header
        ax.add_patch(Rectangle((x0, 0.952), PW, 0.036, facecolor=spec["accent"],
                               edgecolor="none", alpha=0.12))
        ax.plot([x0, x0], [0.952, 0.988], color=spec["accent"], lw=3)
        ax.text(x0 + 0.010, 0.970, spec["model"], ha="left", va="center",
                fontsize=13, fontweight="bold", color=TEXT_D)
        ax.text(x0 + PW - 0.008, 0.970,
                f"layer {spec['layer']} · {spec['corpus']}",
                ha="right", va="center", fontsize=9, color=TEXT_M)

        # column headers
        for (a, b), lab in [(SL_VAR, "share of variance"),
                            (SL_W, "weight in $w$"),
                            (SL_AUC, "standalone AUC"),
                            (SL_POLE, "nearest persona poles (by cosine)")]:
            ax.text(x0 + (a + b) / 2, hdr_y, lab, ha="center", va="center",
                    fontsize=8.5, color=TEXT_M)

        for pc, y in zip(spec["pcs"], row_y):
            ax.plot([x0, x0 + PW], [y + 0.052, y + 0.052], color=GRID, lw=0.8)
            ax.text(x0 + SL_PC[0] + 0.002, y, pc["name"], ha="left",
                    va="center", fontsize=11, fontweight="bold",
                    color=spec["accent"])
            var_bar(ax, x0 + SL_VAR[0], x0 + SL_VAR[1], y, pc["var"], spec["accent"])
            w_bar(ax, x0 + SL_W[0], x0 + SL_W[1], y, pc["w"], spec["accent"])
            auc_gauge(ax, x0 + SL_AUC[0], x0 + SL_AUC[1], y, pc["auc"],
                      spec["accent"], ci=pc["ci"], ref=spec["assistant_auc"],
                      ticks=(pc is spec["pcs"][-1]))
            poles_block(ax, x0 + SL_POLE[0], x0 + SL_POLE[1], y, pc,
                        fs=9, sub_fs=6.0)

        # footer row: assistant axis reference (Llama only)
        yf = 0.462
        ax.plot([x0, x0 + PW], [yf + 0.028, yf + 0.028], color=GRID, lw=0.8)
        if spec["assistant_auc"]:
            ax.text(x0 + 0.002, yf, "assistant axis (PC1 of persona space)",
                    ha="left", va="center", fontsize=8.5, color=TEXT_M,
                    style="italic")
            auc_gauge(ax, x0 + SL_AUC[0], x0 + SL_AUC[1], yf,
                      spec["assistant_auc"], TEXT_L)
        else:
            ax.text(x0 + 0.002, yf,
                    "top-4 PCs: 57.2% of variance · 89% of $\\|w\\|^2$",
                    ha="left", va="center", fontsize=8.5, color=TEXT_M,
                    style="italic")

    # ---------------- steering band ----------------
    ax.plot([0.010, 0.985], [0.425, 0.425], color=SPINE, lw=1.2)
    ax.text(0.010, 0.405,
            "Steering a benign prompt along one component "
            "(Llama-3.1-8B, layer 16, greedy; coefficient × per-token norm)",
            ha="left", va="center", fontsize=10.5, fontweight="bold",
            color=TEXT_D)

    ladders = [(STEER[0], 0.010), (STEER[1], 0.505)]
    LW_ = 0.475
    for st, x0 in ladders:
        ax.text(x0, 0.372, f"{st['pc']}:", ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=C_LLAMA)
        ax.text(x0 + 0.035, 0.372, f"“{st['prompt']}”",
                ha="left", va="center", fontsize=8.5, color=TEXT_M,
                style="italic")
        rows = [(+1, f"+{st['cmax']:.1f}", st["pos_pole"], 0.318),
                (0, "0.0", "unsteered", 0.235),
                (-1, f"−{st['cmax']:.1f}", st["neg_pole"], 0.152)]
        for k, cff, pole, y in rows:
            chip_c = C_LLAMA if k != 0 else TEXT_L
            ax.add_patch(FancyBboxPatch(
                (x0, y - 0.017), 0.052, 0.038,
                boxstyle="round,pad=0,rounding_size=0.006",
                facecolor=chip_c, alpha=0.12 if k == 0 else 0.16,
                edgecolor="none"))
            ax.text(x0 + 0.026, y + 0.007, cff, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=chip_c)
            ax.text(x0 + 0.026, y - 0.009, pole, ha="center", va="center",
                    fontsize=6.8, color=chip_c)
            ax.text(x0 + 0.062, y, wrap(f"“{st['quotes'][k]}”", 78),
                    ha="left", va="center", fontsize=8.2, color=TEXT_D,
                    linespacing=1.25)

    ax.text(0.010, 0.095,
            "Each row: one unsupervised principal component of the residual-stream activations over jailbroken and refused responses.  Share of variance: fraction of\n"
            "activation variance the PC explains.  Weight in $w$: the PC's coefficient in the unit-norm jailbreak classifier direction.  Standalone AUC: jailbroken-vs-refused\n"
            "detection using the single PC projection (dashed marker: assistant axis of Chen et al., 0.64).  Poles: nearest persona trait vectors by cosine.",
            ha="left", va="top", fontsize=8, color=TEXT_M, linespacing=1.5)

    fig.savefig(OUT / "pca_megafig_v1.png", dpi=160)
    plt.close(fig)


# ================================================================ V2
# Narrative flow: decomposition stacked bars -> persona-axis rows -> steering dial.

def v2():
    fig, ax = canvas(13.2, 9.4)

    # ---- top: the decomposition story
    ax.text(0.015, 0.972, "1", fontsize=13, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc=TEXT_D, ec="none"))
    ax.text(0.035, 0.972,
            "The jailbreak direction lives in a 4-component PCA subspace",
            fontsize=12.5, fontweight="bold", color=TEXT_D, ha="left",
            va="center")
    ax.text(0.035, 0.947,
            "PCA of residual-stream activations over jailbroken + refused responses (no labels); "
            "$w$ = unit normal of the linear jailbreak classifier, decomposed on the PCs (squared weights).",
            fontsize=8.7, color=TEXT_M, ha="left", va="center")

    bx0, bx1 = 0.115, 0.640
    for spec, y in [(LLAMA, 0.905), (GEMMA, 0.852)]:
        ax.text(bx0 - 0.008, y, spec["model"], ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=spec["accent"])
        x = bx0
        segs = spec["w2"] + [1 - sum(spec["w2"])]
        cols = C_PC + [C_RESID]
        labs = ["PC1", "PC2", "PC3", "PC4", "rest"]
        for s, c, lab in zip(segs, cols, labs):
            wseg = (bx1 - bx0) * s
            ax.add_patch(Rectangle((x, y - 0.0155), wseg - 0.0012, 0.031,
                                   facecolor=c, edgecolor="none"))
            if s > 0.055:
                ax.text(x + wseg / 2, y, f"{lab} {s * 100:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
            elif s > 0.028:
                ax.text(x + wseg / 2, y, f"{s * 100:.0f}", ha="center",
                        va="center", fontsize=7, color="white",
                        fontweight="bold")
            else:
                ax.text(x + wseg / 2, y + 0.024, f"{lab} {s * 100:.0f}%",
                        ha="center", va="center", fontsize=6.5, color=TEXT_M)
                ax.plot([x + wseg / 2, x + wseg / 2],
                        [y + 0.017, y + 0.0165], color=TEXT_M, lw=0.6)
            x += wseg
    ax.text(0.655, 0.905, "top-4 PCs carry 97% of $\\|w\\|^2$", fontsize=8.5,
            color=TEXT_M, ha="left", va="center")
    ax.text(0.655, 0.852, "89% — mostly PC1 + PC4", fontsize=8.5,
            color=TEXT_M, ha="left", va="center")

    # ---- middle: each PC is a persona axis
    ax.text(0.015, 0.800, "2", fontsize=13, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc=TEXT_D, ec="none"))
    ax.text(0.035, 0.800, "Each component is a persona axis, and it predicts jailbreaks on its own",
            fontsize=12.5, fontweight="bold", color=TEXT_D, ha="left",
            va="center")
    ax.text(0.035, 0.776,
            "Poles: nearest persona trait vectors by cosine.  Badge: standalone jailbroken-vs-refused AUC of the "
            "single PC projection (chance 0.5; assistant axis of Chen et al.: 0.64 on Llama).",
            fontsize=8.7, color=TEXT_M, ha="left", va="center")

    halves = [(LLAMA, 0.105), (GEMMA, 0.560)]
    HW = 0.425
    for spec, hx in halves:
        ax.text(hx + HW / 2, 0.745, spec["model"], ha="center", va="center",
                fontsize=10, fontweight="bold", color=spec["accent"])

    row_y = [0.700, 0.615, 0.530, 0.445]
    for i, y in enumerate(row_y):
        ax.add_patch(Rectangle((0.015, y - 0.012), 0.014, 0.024,
                               facecolor=C_PC[i], edgecolor="none"))
        ax.text(0.037, y, f"PC{i + 1}", ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=TEXT_D)
        for spec, hx in halves:
            pc = spec["pcs"][i]
            xm = hx + HW / 2
            # axis line with poles
            ax.annotate("", xy=(xm + 0.075, y - 0.014), xytext=(xm - 0.075, y - 0.014),
                        arrowprops=dict(arrowstyle="<|-|>", color=TEXT_M,
                                        lw=1.1, mutation_scale=9))
            ax.text(xm - 0.080, y - 0.014, pc["neg"][0][0], ha="right",
                    va="center", fontsize=9.5, fontweight="bold", color=TEXT_D)
            ax.text(xm + 0.080, y - 0.014, pc["pos"][0][0], ha="left",
                    va="center", fontsize=9.5, fontweight="bold", color=TEXT_D)
            sub_n = " · ".join(t for t, _ in pc["neg"][1:3])
            sub_p = " · ".join(t for t, _ in pc["pos"][1:3])
            ax.text(xm - 0.080, y - 0.037, sub_n, ha="right", va="center",
                    fontsize=7, color=TEXT_M)
            ax.text(xm + 0.080, y - 0.037, sub_p, ha="left", va="center",
                    fontsize=7, color=TEXT_M)
            # AUC + var badges above the line
            ax.text(xm - 0.075, y + 0.020,
                    f"var {pc['var']:.1f}%   $w$ {pc['w']:+.2f}",
                    ha="left", va="center", fontsize=7.5, color=TEXT_M)
            ax.scatter([xm + 0.052], [y + 0.020], s=34, color=spec["accent"],
                       edgecolor="white", linewidth=0.8, zorder=5)
            ax.text(xm + 0.060, y + 0.020, f"AUC {pc['auc']:.2f}", ha="left",
                    va="center", fontsize=8.3, fontweight="bold", color=TEXT_D)

    # ---- bottom: steering dial
    ax.text(0.015, 0.375, "3", fontsize=13, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc=TEXT_D, ec="none"))
    ax.text(0.035, 0.375, "Steering along a component moves the model along that persona",
            fontsize=12.5, fontweight="bold", color=TEXT_D, ha="left",
            va="center")
    ax.text(0.035, 0.351,
            "Greedy continuations of a benign prompt while adding coeff × per-token norm of PC1 / PC2 "
            "to the layer-16 residual stream (Llama-3.1-8B).",
            fontsize=8.7, color=TEXT_M, ha="left", va="center")

    for st, y0 in [(STEER[0], 0.288), (STEER[1], 0.138)]:
        dial_x0, dial_x1 = 0.075, 0.930
        i_pc = 0 if st["pc"] == "PC1" else 1
        ax.add_patch(Rectangle((0.015, y0 - 0.010), 0.011, 0.020,
                               facecolor=C_PC[i_pc], edgecolor="none"))
        ax.text(0.031, y0, st["pc"], ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=TEXT_D)
        ax.annotate("", xy=(dial_x1, y0), xytext=(dial_x0, y0),
                    arrowprops=dict(arrowstyle="<|-|>", color=SPINE, lw=1.4,
                                    mutation_scale=11))
        ax.text(dial_x0 - 0.004, y0 - 0.020, st["neg_pole"], ha="left",
                va="center", fontsize=8.5, fontweight="bold", color=TEXT_M)
        ax.text(dial_x1 + 0.004, y0 - 0.020, st["pos_pole"], ha="right",
                va="center", fontsize=8.5, fontweight="bold", color=TEXT_M)
        ax.text(0.075, y0 + 0.046, f"“{st['prompt']}”", ha="left",
                va="center", fontsize=8.3, color=TEXT_M, style="italic")
        for k, xc in [(-1, 0.215), (0, 0.500), (+1, 0.790)]:
            cff = {1: f"+{st['cmax']:.1f}", 0: "0.0",
                   -1: f"−{st['cmax']:.1f}"}[k]
            ax.plot([xc, xc], [y0 - 0.008, y0 + 0.008], color=TEXT_M, lw=1.4)
            ax.text(xc, y0 + 0.016, cff, ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=TEXT_D)
            ax.add_patch(FancyBboxPatch(
                (xc - 0.13, y0 - 0.085), 0.26, 0.058,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor="#FAFAFA", edgecolor=SPINE, linewidth=0.8))
            ax.text(xc, y0 - 0.056, wrap(f"“{st['quotes'][k]}”", 52),
                    ha="center", va="center", fontsize=7.4, color=TEXT_D,
                    linespacing=1.3)

    fig.savefig(OUT / "pca_megafig_v2.png", dpi=160)
    plt.close(fig)


# ================================================================ V3
# Portrait: models stacked; steering quotes inline per Llama PC row.

def v3():
    fig, ax = canvas(9.0, 11.8)

    SL_VAR = (0.075, 0.155)
    SL_W   = (0.185, 0.280)
    SL_AUC = (0.310, 0.400)
    X_POLE = (0.415, 0.630)
    X_QT   = 0.645

    def clip(s, n):
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"

    def block_header(spec, y):
        ax.add_patch(Rectangle((0.015, y - 0.014), 0.970, 0.028,
                               facecolor=spec["accent"], edgecolor="none",
                               alpha=0.12))
        ax.plot([0.015, 0.015], [y - 0.014, y + 0.014], color=spec["accent"],
                lw=3)
        ax.text(0.025, y, spec["model"], ha="left", va="center", fontsize=12,
                fontweight="bold", color=TEXT_D)
        ax.text(0.980, y, f"layer {spec['layer']} · {spec['corpus']}",
                ha="right", va="center", fontsize=8.5, color=TEXT_M)

    def col_headers(y, with_steer):
        for (a, b), lab in [(SL_VAR, "variance"), (SL_W, "weight in $w$"),
                            (SL_AUC, "standalone AUC")]:
            ax.text((a + b) / 2, y, lab, ha="center", va="center",
                    fontsize=7.5, color=TEXT_M)
        ax.text(sum(X_POLE) / 2, y, "persona poles", ha="center", va="center",
                fontsize=7.5, color=TEXT_M)
        if with_steer:
            ax.text(X_QT, y, "steered continuation (+coeff → positive pole)",
                    ha="left", va="center", fontsize=7.5, color=TEXT_M)

    # ---------- Llama block with steering quotes
    block_header(LLAMA, 0.972)
    col_headers(0.943, True)
    row_y = [0.885, 0.775, 0.665, 0.555]
    for pc, st, y in zip(LLAMA["pcs"], STEER, row_y):
        ax.plot([0.015, 0.985], [y + 0.052, y + 0.052], color=GRID, lw=0.8)
        ax.text(0.020, y, pc["name"], ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=C_LLAMA)
        var_bar(ax, *SL_VAR, y, pc["var"], C_LLAMA, fs=7.5)
        w_bar(ax, *SL_W, y, pc["w"], C_LLAMA, fs=7.5)
        auc_gauge(ax, *SL_AUC, y, pc["auc"], C_LLAMA, ref=LLAMA["assistant_auc"],
                  fs=7.5, ticks=(pc is LLAMA["pcs"][-1]))
        poles_block(ax, *X_POLE, y, pc, fs=8.5, sub_fs=6.0)
        # steering quotes: extremes only
        ax.text(X_QT, y + 0.042, f"“{clip(st['prompt'], 56)}”", ha="left",
                va="center", fontsize=6.8, color=TEXT_M, style="italic")
        for k, dy in [(+1, 0.026), (-1, -0.016)]:
            cff = (f"+{st['cmax']:.1f}" if k > 0 else f"−{st['cmax']:.1f}")
            ax.text(X_QT, y + dy, cff, ha="left", va="top", fontsize=7.5,
                    fontweight="bold", color=C_LLAMA)
            ax.text(X_QT + 0.036, y + dy,
                    wrap(f"“{clip(st['quotes'][k], 92)}”", 52), ha="left",
                    va="top", fontsize=6.8, color=TEXT_D, linespacing=1.3)
    # assistant axis note
    ax.text(0.020, 0.495,
            "assistant axis (PC1 of persona space): standalone AUC 0.64 — dashed marker on the gauges",
            ha="left", va="center", fontsize=7.5, color=TEXT_M, style="italic")

    # ---------- Gemma block, poles expanded (no steering runs)
    block_header(GEMMA, 0.455)
    col_headers(0.426, False)
    ax.text(X_QT, 0.426, "nearest traits per pole (cos)", ha="left",
            va="center", fontsize=7.5, color=TEXT_M)
    row_yg = [0.375, 0.285, 0.195, 0.105]
    for pc, y in zip(GEMMA["pcs"], row_yg):
        ax.plot([0.015, 0.985], [y + 0.042, y + 0.042], color=GRID, lw=0.8)
        ax.text(0.020, y, pc["name"], ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=C_GEMMA)
        var_bar(ax, *SL_VAR, y, pc["var"], C_GEMMA, fs=7.5)
        w_bar(ax, *SL_W, y, pc["w"], C_GEMMA, fs=7.5)
        auc_gauge(ax, *SL_AUC, y, pc["auc"], C_GEMMA, ci=pc["ci"], fs=7.5,
                  ticks=(pc is GEMMA["pcs"][-1]))
        poles_block(ax, *X_POLE, y, pc, fs=8.5, subs=False)
        neg = "  ".join(f"{t} ({c:.2f})" for t, c in pc["neg"])
        pos = "  ".join(f"{t} ({c:.2f})" for t, c in pc["pos"])
        ax.text(X_QT, y + 0.014, "− " + neg, ha="left", va="center",
                fontsize=7, color=TEXT_M)
        ax.text(X_QT, y - 0.014, "+ " + pos, ha="left", va="center",
                fontsize=7, color=TEXT_M)

    ax.text(0.015, 0.040,
            "Unsupervised PCA of residual-stream activations over jailbroken and refused responses. Weight in $w$: the PC's coefficient in\n"
            "the unit-norm jailbreak classifier direction. Standalone AUC: jailbroken-vs-refused detection from the single PC projection\n"
            "(ticks 0.5 / 0.7 / 0.9). Steering: greedy continuation while adding coeff × per-token norm of the PC to the residual stream.",
            ha="left", va="top", fontsize=7.5, color=TEXT_M, linespacing=1.45)

    fig.savefig(OUT / "pca_megafig_v3.png", dpi=160)
    plt.close(fig)


# ================================================================ V4 / V5
# Assistant-axis style: succinct typographic table (Table-1 style), optionally
# with a compact steering ladder (their Table-3 style).

# full top-5 pole lists from tables/pca_poles_{llama,gemma}.tex
POLES5 = {
    "llama": [
        (["diplomatic", "calm", "methodical", "solemn", "reserved"],
         ["edgy", "sycophantic", "goofy", "grandiose", "mischievous"]),
        (["evil", "condescending", "misanthropic", "bitter", "elitist"],
         ["principled", "descriptive", "conciliatory", "conscientious", "egalitarian"]),
        (["reserved", "dispassionate", "detached", "solemn", "serious"],
         ["introspective", "witty", "entertaining", "adaptable", "gregarious"]),
        (["casual", "hostile", "evil", "improvisational", "sassy"],
         ["ritualistic", "formal", "conscientious", "reverent", "technical"]),
    ],
    "gemma": [
        (["generalist", "strategic", "methodical", "erudite", "structuralist"],
         ["impatient", "competitive", "pessimistic", "pacifist", "submissive"]),
        (["accessible", "systems thinker", "meticulous", "methodical", "practical"],
         ["enigmatic", "obsessive", "concise", "impatient", "cryptic"]),
        (["exploratory", "systems thinker", "cautious", "inclusive", "adaptable"],
         ["impulsive", "nonchalant", "essentialist", "obsessive", "serious"]),
        (["dominant", "structuralist", "critical", "conceptual", "systems thinker"],
         ["concise", "submissive", "socratic", "humble", "impatient"]),
    ],
}


def _pole_line(ax, x, y, traits, shared, fs=9):
    """Comma-separated trait list; traits in `shared` set in bold."""
    for i, t in enumerate(traits):
        s = t + ("," if i < len(traits) - 1 else "")
        txt = ax.text(x, y, s, ha="left", va="center", fontsize=fs,
                      color=TEXT_D,
                      fontweight="bold" if t in shared else "normal")
        # advance by measured width
        ax.figure.canvas.draw()
        bb = txt.get_window_extent()
        x = ax.transData.inverted().transform((bb.x1, 0))[0] + 0.006


def _table_block(ax, y_top, pitch):
    """Shared assistant-axis-style table body; returns y after last block."""
    XL, XG = 0.095, 0.555          # model column starts
    # header
    ax.text(XL, 0.965, "Llama-3.1-8B", ha="left", va="center", fontsize=11,
            fontweight="bold", color=C_LLAMA)
    ax.text(0.505, 0.9645, "layer 16 · HarmBench", ha="right",
            va="center", fontsize=8, color=TEXT_M)
    ax.text(XG, 0.965, "Gemma-4-31B", ha="left", va="center", fontsize=11,
            fontweight="bold", color=C_GEMMA)
    ax.text(0.98, 0.9645, "layer 30 · HarmBench + WildJB", ha="right",
            va="center", fontsize=8, color=TEXT_M)
    ax.plot([0.02, 0.98], [0.945, 0.945], color=TEXT_D, lw=1.2)

    for i in range(4):
        yt = y_top - i * pitch
        ax.text(0.02, yt, f"PC{i + 1}", ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=TEXT_D)
        for key, spec, x0 in [("llama", LLAMA, XL), ("gemma", GEMMA, XG)]:
            pc = spec["pcs"][i]
            other = POLES5["gemma" if key == "llama" else "llama"][i]
            shared_n = set(POLES5[key][i][0]) & set(other[0])
            shared_p = set(POLES5[key][i][1]) & set(other[1])
            ci = f" ± {pc['ci']:.2f}" if pc["ci"] else ""
            ax.text(x0, yt, f"{pc['var']:.1f}% var  ·  weight in $w$ "
                    f"{pc['w']:+.2f}  ·  standalone AUC {pc['auc']:.2f}{ci}",
                    ha="left", va="center", fontsize=8, color=TEXT_M)
            ax.text(x0, yt - 0.052, "−", ha="center", va="center",
                    fontsize=9.5, color=TEXT_M, fontweight="bold")
            _pole_line(ax, x0 + 0.014, yt - 0.052, POLES5[key][i][0], shared_n)
            ax.text(x0, yt - 0.104, "+", ha="center", va="center",
                    fontsize=9.5, color=TEXT_M, fontweight="bold")
            _pole_line(ax, x0 + 0.014, yt - 0.104, POLES5[key][i][1], shared_p)
        if i < 3:
            ax.plot([0.02, 0.98], [yt - pitch + (pitch - 0.104) / 2] * 2,
                    color=GRID, lw=0.8)
    y_end = y_top - 3 * pitch - 0.104
    # assistant-axis reference row
    ax.plot([0.02, 0.98], [y_end - 0.035] * 2, color=GRID, lw=0.8)
    ax.text(0.02, y_end - 0.070,
            "assistant axis (PC1 of persona space; Chen et al.):  "
            "standalone AUC 0.64 on Llama — below every component above",
            ha="left", va="center", fontsize=8.5, color=TEXT_M, style="italic")
    ax.plot([0.02, 0.98], [y_end - 0.100] * 2, color=TEXT_D, lw=1.2)
    return y_end - 0.100


def v4():
    fig, ax = canvas(11.0, 5.9)
    y_end = _table_block(ax, y_top=0.885, pitch=0.20)
    ax.text(0.02, y_end - 0.045,
            "Top five persona trait vectors by cosine at each pole of the top four activation PCs; "
            "bold: trait shared by both models at the same pole. Signs oriented as in the decomposition of $w$.",
            ha="left", va="top", fontsize=8, color=TEXT_M, linespacing=1.4)
    fig.savefig(OUT / "pca_megafig_v4.png", dpi=180)
    plt.close(fig)


def v5():
    fig, ax = canvas(11.0, 9.0)
    y_end = _table_block(ax, y_top=0.925, pitch=0.130)

    def clip(s, n):
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"

    y = y_end - 0.042
    ax.text(0.02, y, "Steering along a component (Llama-3.1-8B, layer 16, "
            "greedy continuation of a benign prompt)",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=TEXT_D)
    y -= 0.034
    for st in STEER[:2]:
        ax.text(0.02, y, st["pc"], ha="left", va="center", fontsize=9,
                fontweight="bold", color=C_LLAMA)
        ax.text(0.062, y, f"“{st['prompt']}”", ha="left", va="center",
                fontsize=8.5, color=TEXT_M, style="italic")
        y -= 0.030
        for k in (+1, 0, -1):
            cff = {1: f"+{st['cmax']:.1f}", 0: " 0.0",
                   -1: f"−{st['cmax']:.1f}"}[k]
            pole = {1: st["pos_pole"], 0: "", -1: st["neg_pole"]}[k]
            ax.text(0.062, y, cff, ha="right", va="center", fontsize=8.5,
                    fontweight="bold", color=C_LLAMA if k else TEXT_L)
            ax.text(0.075, y, f"“{clip(st['quotes'][k], 105)}”", ha="left",
                    va="center", fontsize=8.5, color=TEXT_D)
            if pole:
                ax.text(0.98, y, pole, ha="right", va="center", fontsize=8,
                        color=TEXT_M, style="italic")
            y -= 0.030
        y -= 0.012
    fig.savefig(OUT / "pca_megafig_v5.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    v4()
    v5()
    print("wrote", *(str(OUT / f"pca_megafig_v{i}.png") for i in (4, 5)))
