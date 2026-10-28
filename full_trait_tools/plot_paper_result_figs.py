#!/usr/bin/env python3
"""
plot_paper_result_figs.py

Journal-style restyles of the remaining main-text result figures (previously
presentation-style PNGs from make_presentation_figs.py / make_slides.py):

  1. fig_detection_transfer  - avg transfer AUC per detector (bar)
  2. fig_id_vs_transfer      - in-distribution vs transfer AUC (scatter)
  3. fig_accuracy_vs_cost    - transfer AUC vs added latency (scatter, log x)
  4. fig_layer_sweep_auc     - full-classifier AUC by layer (lines)
  5. fig_attack_refused      - steering vs standard attacks ASR (bar)
  6. fig_persona_readout_probe - fresh per-position probe AUC (line)

Numbers for 1/2/5 are the paper's own tables (tab:detection, tab:attack);
3 reuses the latency benchmark values hardcoded in make_presentation_figs.py;
4 reads all_layers_sweep_summary.csv; 6 reads per_position_probe_recast_auc.npz.

Usage:
  python3 full_trait_tools/plot_paper_result_figs.py \
      [--outdir full_trait_output/plots/paper_figs] [--png]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.transforms import ScaledTranslation

# journal palette (matches the Figure 1/2 scripts); ours = brick, the
# raw-activation counterpart = amber, third-party baselines = gray
C_OURS  = "#A6403A"
C_RAW   = "#C88A2D"
C_BASE  = "#9AA4AD"
C_HEAD  = "#1D5B8C"
GRID    = "#E7E7E7"
SPINE   = "#C9C9C9"
TEXT_M  = "#555555"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def new_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_M, labelsize=9, length=0)
    return fig, ax


def halo(t, lw=2.5):
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])


def save(fig, outdir, name, png):
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", dpi=200, bbox_inches="tight")
    if png:
        fig.savefig(outdir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", outdir / name)


# tab:detection - name: (in-distribution AUC, avg transfer AUC, 95% CI, class)
#
# Rebrand 2026-08-23: the trained trait-projection classifier ("PersonaJB", was
# 0.86 / 0.76 +-.004) is replaced by the direction it defines. The vector is
# Dominic's steering-attack normal, rebuilt by
# jailbreaking_tools/trait_steering_attack_jbb_persona_w.py's own recipe
# (build_proj_features -> fit_classifier -> persona_w: trait projections at
# layer 16, StandardScaler, LogisticRegression(max_iter=2000, C=1.0, seed 42),
# w = (coef/scale) @ T, unit-norm), scored on the five unseen families as a
# plain projection x.w -- which is exactly the classifier's own AUC, since its
# decision function is w.x - const. Identity check: cv_auc 0.96812 against the
# 0.9681309216192938 recorded by the published steering run's summary.json, and
# cosine 1.0000000000 with steering_directions_layer16.pt's w_persona__full.
# Numbers are the published trait row's, unchanged: same config
# (logreg_l2_C10.0_raw), same strict 70/30 behaviour x template split, 50 seeds,
# read from all_traits_sweep_v2/results.json. That is deliberate -- w = T'(beta/
# sigma) is that classifier's own decision normal and AUC(x.w) is its AUC, so at
# matched data and matched C the two are the same measurement and the rebrand
# changes only the name and the latency. Fitting instead on all 3180 pairs (the
# vector the steering attack uses) transfers at .776, but on 2.05x the training
# pairs the baselines get -- the 70/30 product split keeps only pure cells, 1554
# train / 288 test of 3180 -- so it is not comparable and is not used here.
# Per family: GCG .8166 PAIR .7900 PAP .6413 GPTFuzz .7083 PEZ .8416,
# in-distribution .8538; avg transfer .7596 +-.0039 over the 50 seed means.
DET = [
    ("PersonaJB Direction",        0.854, 0.760, 0.004, "ours"),
    # PCA with k matched to the trait count (229 on Llama). Rerun 2026-09-17 with
    # full_trait_output/pca_k_sweep/pca_k_sweep.py --model llama --ks 229 (same
    # fast_transfer / run_transfer_classifier protocol as the published PCA row,
    # 50 seeds): sweep_llama_k229.json. Per family GCG .7592 PAIR .7754 PAP .6463
    # GPTFuzz .6175 PEZ .8058, in-distribution .8313; avg transfer .7208 +-.008.
    # The previous row (0.88 / 0.68 +-.002, labelled "PCA (3)") had run with n_pca=4.
    ("PCA",                        0.831, 0.721, 0.008, "raw"),
    ("Mahalanobis",                0.92, 0.71, 0.006, "base"),
    ("Activation Probe",           0.81, 0.71, 0.010, "raw"),
    ("JLT (SVM)",                  0.91, 0.70, 0.012, "base"),
    # The two directions from the direction-story experiment, scored as detectors
    # (dirs_as_detectors.py / dirs_fig_inputs.py). Identity confirmed by their
    # in-span shares reproducing fig_direction_story: refusal .407 (Llama) /
    # .265 (Gemma), dmu .460 / .326 against the published .482 / .330.
    #   Jailbreak dir = mean(jailbroken) - mean(refused), refit per seed on the
    #     same 70/30 train pool as every trained row (Gemma folds in WJB, as its
    #     other trained rows do) -> seed CI. In-distribution .856 / .818.
    #   Refusal dir = -(Arditi diff-of-means), layer 16 pos -1 for Llama, and for
    #     Gemma rebuilt from moskvore/gemma_refusal_dir class means at
    #     hidden_states[31] pos -1. Estimated on Arditi's harmful/harmless sets,
    #     so HarmBench and all five families are held out and there is no pool to
    #     match; being deterministic it carries a test-set bootstrap CI, not a
    #     seed CI. In-distribution .831 / .800. Gemma's pos -5 variant is not
    #     shown: position -5 is a template token there (in-span share .708, AUCs
    #     below chance), so Arditi's causal preference for -5 does not carry over.
    ("Jailbreak Direction",              0.856, 0.693, 0.003, "raw"),
    ("JBShield",                   0.85, 0.69, 0.003, "base"),
    ("Refusal Direction",                0.831, 0.670, 0.023, "raw"),
    ("JLT (RF)",                   0.89, 0.64, 0.014, "base"),
    # GradSafe CI: test-set bootstrap over per-prompt scores regenerated with the
    # pre-sliced fast scorer; all five families reproduce the table (0.58/0.62/
    # 0.60/0.52/0.76) and the mean 0.6158 -> 0.62
    ("GradSafe",                   0.95, 0.62, 0.027, "base"),
    ("Llama Guard",                0.69, 0.61, 0.027, "base"),
    # Verbalized CIs: 1000-resample test-set bootstrap over per-example labels
    # recovered by re-judging the stored meta-responses with gpt-4.1-mini
    # (re-judged per-family AUCs reproduce the stored ones to <=0.007)
    ("Verbalized (no CoT)",        0.58, 0.55, 0.013, "base"),
    ("Verbalized (CoT)",           0.58, 0.52, 0.008, "base"),
    ("Perplexity",                 0.65, 0.51, 0.034, "base"),
]
CLASS_COLOR = {"ours": C_OURS, "raw": C_RAW, "base": C_BASE}

# tab:detection-gemma avg transfer over all five unseen families, with 95% CI of
# the seed mean (1.96*std/sqrt(n)) computed from the per-seed AUCs (50 seeds;
# JLT 20). Updated 2026-08-23: the GCG-on-Gemma detector runs exist for every
# method, so the average is no longer taken over four families. Each row is read
# from the block of the *_gemma_gcg_* run whose four non-GCG families reproduce
# tab:detection-gemma (max per-family deviation 0.005, i.e. rounding):
#   PersonaJB        all_traits_sweep_v2_gemma_gcg_all_traits  logreg_l2_C1.0_bal
#   Activation Probe all_traits_sweep_v2_gemma_gcg_raw         logreg_l2_C1.0_bal
#   PCA              all_traits_sweep_v2_gemma_gcg_pca3        logreg_l2_C10.0_bal
#   Mahalanobis      baselines_contrastive_geometry_gemma_gcg_hbwjb  methods/mcd
#   JLT (SVM|RF)     baselines_jlt_hidden_tensor_gemma_gcg_hbwjb     models/pca_*
#   JBShield         baselines_jbshield_fjd_gemma_gcg_hbwjb   jbshield_mean_direction
#   GradSafe         baselines_gradsafe_gemma_bal             results
#   Llama Guard      baselines_all_attacks_gemma_gcg          /
#   Verbalized       baselines_self_exam_gemma_gcg (GCG) + baselines_all_attacks_gemma
#   Perplexity       baselines_harmbench_ppl_smoothllm_gemma_gcg  perplexity_nll
# The five deterministic rows store no per-example GCG scores, so their CI is the
# earlier four-family one carried over, even though the point estimate moved.
DET_GEMMA = {
    # Rebrand 2026-08-23, see DET. Same config as the Llama row
    # (logreg_l2_C10.0_raw) on the same 70/30 split, 50 seeds, from
    # all_traits_sweep_v2_gemma_gcg_all_traits/results.json. This moves the row
    # off the paper's previous Gemma config (logreg_l2_C1.0_bal, .7281 +-.0034):
    # the two models were configured differently, and matching them is what
    # raises Gemma here. Per family GCG .7383 PAIR .6782 PAP .6937 GPTFuzz .7590
    # PEZ .8402, in-distribution .8773; avg transfer .7419 +-.0025.
    "PersonaJB Direction":      (0.742, 0.003),
    "Mahalanobis":              (0.696, 0.001),
    "Activation Probe":         (0.651, 0.006),
    "JLT (SVM)":                (0.675, 0.002),
    "JBShield":                 (0.674, 0.002),
    # PCA with k = 240 trait directions (pca_k_sweep.py --model gemma --ks 240;
    # sweep_v2 protocol, logreg_l2_C1.0_bal, HB+WJB train, 50 seeds):
    # sweep_gemma_k240.json. GCG .7327 PAIR .7169 PAP .6759 GPTFuzz .7048 PEZ
    # .6406, in-distribution .7854; avg transfer .6942 +-.003. Was PCA(3): .623.
    "PCA":                      (0.694, 0.003),
    "JLT (RF)":                 (0.687, 0.004),
    "Jailbreak Direction":            (0.677, 0.001),   # see DET
    "Refusal Direction":              (0.662, 0.024),   # bootstrap CI
    # WJB-calibrated arm, finished 2026-08-21: PEZ (the family that stalled the
    # original run) came in at 0.912, and GCG at 0.749. Evaluated on balanced
    # slices (every positive + a matched negative), unlike the full-pool rows
    # -> hatched in the figure, footnote in the caption.
    "GradSafe":                 (0.702, 0.028),   # CI: 4-family, carried over
    "Llama Guard":              (0.566, 0.034),   # CI carried over
    "Verbalized (no CoT)":      (0.516, 0.014),   # CI carried over
    "Verbalized (CoT)":         (0.522, 0.009),   # CI carried over
    "Perplexity":               (0.505, 0.029),   # CI carried over
}
C_LLAMA, C_GEMMA = C_OURS, C_RAW   # model colors, as in the Fig. 1 dotplot
# Reserved for Gemma rows that are genuinely not comparable to the rest of the
# panel. Empty since 2026-08-23: GradSafe used to be faded here for being
# "WJB-calibrated and scored on balanced slices", and neither holds up. Every
# trained Gemma row folds in WildJailbreak -- GradSafe uses the least of it, 20+20
# calibration prompts against the 2000 rows in the classifier pools -- and
# balance_positives() is pos + rng.sample(neg, len(pos)), a uniform subsample of
# the same pool, so P(score(pos) > score(neg)) is estimated without bias. What
# the slice costs is variance, which its +-.028 (the widest in the panel) already
# shows, plus one real difference confined to the in-distribution column: its
# HarmBench cell takes all 63 positives from the whole 3180-pair pool rather than
# the held-out 288-pair cell. No leakage -- it never trains on HarmBench.
GEMMA_BALANCED = set()
BAL_ALPHA = 0.22   # bar fill
BAL_INK   = 0.45   # its CI whisker and value label, faded with it

CHANCE = 0.5


def fig_detection_transfer(outdir, png):
    fig, ax = new_ax((5.6, 4.0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ys = np.arange(len(DET))[::-1] * 1.0
    off, h = 0.19, 0.34

    def whisker(v, ci, y, alpha=1.0):
        """CI bar, clipped at chance: these AUCs are reported as max(auc, 1-auc),
        so an interval cannot meaningfully extend below 0.5."""
        if not ci:
            return v
        lo, hi = max(v - ci, CHANCE), v + ci
        ax.plot([lo, hi], [y, y], color="#4A4A4A", lw=1.1, alpha=alpha,
                solid_capstyle="butt" if lo == CHANCE else "round", zorder=4)
        return hi

    for y, (name, _, tr, ci, cls) in zip(ys, DET):
        ge = DET_GEMMA[name]
        bold = "bold" if cls == "ours" else "normal"
        # bars grow from chance, which is also the left edge of the axis
        ax.barh(y + off, tr - CHANCE, left=CHANCE, height=h, color=C_LLAMA,
                zorder=3)
        right = whisker(tr, ci, y + off)
        ax.text(right + 0.004, y + off, f"{tr:.2f}", va="center",
                ha="left", fontsize=7.5, color=TEXT_M, fontweight=bold)
        if ge is None:
            ax.text(0.503, y - off, "n/a", va="center", ha="left",
                    fontsize=7.5, color=TEXT_M, style="italic")
            continue
        ge_v, ge_ci = ge
        bal = name in GEMMA_BALANCED
        ax.barh(y - off, ge_v - CHANCE, left=CHANCE, height=h, color=C_GEMMA,
                zorder=3, alpha=BAL_ALPHA if bal else 1.0,
                edgecolor=C_GEMMA if bal else "none",
                linewidth=0.9 if bal else 0.0)
        right = whisker(ge_v, ge_ci, y - off, alpha=BAL_INK if bal else 1.0)
        ax.text(right + 0.004, y - off, f"{ge_v:.2f}",
                va="center", ha="left", fontsize=7.5, color=TEXT_M,
                fontweight=bold, alpha=BAL_INK if bal else 1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in DET], fontsize=9)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(CHANCE, 0.80)
    ax.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    ax.set_ylim(-1.0, len(DET) - 0.4)
    # wording kept identical to fig_accuracy_vs_cost's y axis (same quantity)
    ax.set_xlabel("Avg. OOD jailbreak detection AUC", fontsize=10, color=TEXT_M)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C_LLAMA, C_GEMMA)]
    leg = ax.legend(handles, ["Llama-3.1-8B", "Gemma-4-31B"],
                    loc="lower right", fontsize=8.5, framealpha=0.95,
                    edgecolor=SPINE, borderpad=0.5)
    leg.get_frame().set_facecolor("white")
    save(fig, outdir, "fig_detection_transfer", png)


def fig_detection_transfer_transposed(outdir, png):
    """Transposed (vertical-bar) form of the main-text detection panel: same
    numbers, colours, whiskers and value labels, roughly half the page height.
    Sized at exactly \\linewidth (5.5in) so nothing is downscaled in the paper,
    and the legend sits in the free upper-right corner above the low-AUC baselines.
    Value labels drop the leading zero and are set vertically: with 14 methods a
    bar is 0.107in wide, and even ".76" at 6pt does not fit flat (the check below
    caught three touching pairs), whereas a rotated label needs only its font
    height. The function checks its own rendered label boxes and warns if any two
    touch."""
    fig, ax = new_ax((5.5, 1.95))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xs = np.arange(len(DET))
    off, width = 0.185, 0.34

    def whisker(v, ci, x, alpha=1.0):
        """CI bar, clipped at chance: these AUCs are reported as max(auc, 1-auc),
        so an interval cannot meaningfully extend below 0.5."""
        lo, hi = max(v - ci, CHANCE), v + ci
        ax.plot([x, x], [lo, hi], color="#4A4A4A", lw=1.0, alpha=alpha,
                solid_capstyle="butt" if lo == CHANCE else "round", zorder=4)
        return hi

    labels = []
    for x, (name, _, tr, ci, cls) in zip(xs, DET):
        ge_v, ge_ci = DET_GEMMA[name]
        llama_x, gemma_x = x - off, x + off
        bold = "bold" if cls == "ours" else "normal"

        # bars grow from chance, which is also the bottom of the axis
        ax.bar(llama_x, tr - CHANCE, bottom=CHANCE, width=width,
               color=C_LLAMA, zorder=3)
        llama_hi = whisker(tr, ci, llama_x)

        bal = name in GEMMA_BALANCED
        ax.bar(gemma_x, ge_v - CHANCE, bottom=CHANCE, width=width,
               color=C_GEMMA, zorder=3,
               alpha=BAL_ALPHA if bal else 1.0,
               edgecolor=C_GEMMA if bal else "none",
               linewidth=0.9 if bal else 0.0)
        gemma_hi = whisker(ge_v, ge_ci, gemma_x,
                           alpha=BAL_INK if bal else 1.0)

        # 14 methods x 2 models leaves each bar ~0.13in: flat ".76" labels no
        # longer fit side by side (the check below caught .67/.66, .52/.52 and
        # .51/.51 touching), so they are set vertically -- a rotated label needs
        # only its font height horizontally and the column never collides.
        labels.append(ax.text(
            llama_x, llama_hi + 0.006, f"{tr:.2f}".lstrip("0"), ha="center",
            va="bottom", rotation=90, fontsize=6.0, color=TEXT_M,
            fontweight=bold))
        labels.append(ax.text(
            gemma_x, gemma_hi + 0.006, f"{ge_v:.2f}".lstrip("0"), ha="center",
            va="bottom", rotation=90, fontsize=6.0, color=TEXT_M,
            fontweight=bold, alpha=BAL_INK if bal else 1.0))

    ax.set_xticks(xs)
    tick_labels = ax.set_xticklabels(
        # two lines for the long name: rotated 52 deg, one line juts far below
        [{"PersonaJB Direction": "PersonaJB\nDirection",
          "Jailbreak Direction": "Jailbreak\nDirection",
          "Refusal Direction": "Refusal\nDirection",
          # the two longest single-line labels set the figure's bottom extent
          "Verbalized (no CoT)": "Verbalized\n(no CoT)",
          "Verbalized (CoT)": "Verbalized\n(CoT)",
          "Activation Probe": "Activation\nProbe"}.get(d[0], d[0])
         for d in DET],
        rotation=52, ha="right", rotation_mode="anchor", fontsize=7.7)
    tick_labels[0].set_fontweight("bold")
    # A two-line tick label leans its second line into the next label's space at
    # 52 deg, so nudge those left to restore the gap on the right. Their own left
    # neighbours are short single-line labels (or the axis edge), so the shift
    # costs nothing there.
    for t in tick_labels:
        if "\n" in t.get_text():
            t.set_transform(t.get_transform()
                            + ScaledTranslation(-5 / 72, 0, fig.dpi_scale_trans))
    ax.set_xlim(-0.62, len(DET) - 0.38)
    ax.set_ylim(CHANCE, 0.825)
    ax.set_yticks([0.50, 0.60, 0.70, 0.80])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    # wording kept identical to fig_accuracy_vs_cost's y axis (same quantity)
    ax.set_ylabel("Avg. OOD jailbreak\ndetection AUC", fontsize=8.5,
                  color=TEXT_M)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C_LLAMA, C_GEMMA)]
    leg = ax.legend(handles, ["Llama-3.1-8B", "Gemma-4-31B"],
                    loc="upper right", ncol=1,
                    fontsize=7.0, framealpha=0.95, edgecolor=SPINE,
                    borderpad=0.45, handlelength=1.3, columnspacing=1.0,
                    handletextpad=0.5)
    leg.get_frame().set_facecolor("white")

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r).expanded(1.08, 1.08) for t in labels]
    clashes = [(labels[i].get_text(), labels[j].get_text())
               for i in range(len(boxes)) for j in range(i + 1, len(boxes))
               if boxes[i].overlaps(boxes[j])]
    if clashes:
        print(f"  WARNING: {len(clashes)} value labels overlap: {clashes}")

    save(fig, outdir, "fig_detection_transfer_transposed", png)


def fig_id_vs_transfer(outdir, png):
    fig, ax = new_ax((5.4, 3.6))
    ax.grid(True, color=GRID, linewidth=0.8)
    lo, hi = 0.45, 1.0
    ax.plot([lo, hi], [lo, hi], ls="--", color="#B0B0B0", lw=1.1, zorder=1)
    t = ax.text(0.62, 0.645, "transfer = in-distribution", fontsize=7.5,
                color=TEXT_M, rotation=38, ha="center", va="bottom",
                rotation_mode="anchor")
    halo(t)
    offs = {
        "PersonaJB Direction":    (7, 6, "left"),
        "Activation Probe":         (-6, 6, "right"),
        "Jailbreak Direction": (7, 4, "left"), "Refusal Direction": (-7, 4, "right"),
        "PCA":                 (0, 9, "center"),
        "Mahalanobis": (0, 8, "center"),
        "JLT (SVM)":               (7, -10, "left"),
        "JBShield":                (0, -11, "center"),
        "JLT (RF)":                (7, 0, "left"),
        "GradSafe":                (2, -12, "center"),
        "Llama Guard":             (7, -3, "left"),
        "Verbalized (no CoT)":     (7, 2, "left"),
        "Verbalized (CoT)":        (-7, -10, "right"),
        "Perplexity":              (7, -3, "left"),
    }
    for name, idv, tr, _, cls in DET:
        c = CLASS_COLOR[cls]
        if cls == "ours":
            ax.scatter(idv, tr, s=170, marker="*", color=c, zorder=5,
                       edgecolors="white", linewidths=0.7)
        else:
            ax.scatter(idv, tr, s=42 if cls == "raw" else 30, color=c,
                       zorder=4, edgecolors="white", linewidths=0.5)
        dx, dy, ha = offs[name]
        t = ax.annotate(name, (idv, tr), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7.5,
                        color=c if cls != "base" else TEXT_M,
                        fontweight="bold" if cls != "base" else "normal")
        halo(t)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.47, 0.83)
    ax.set_xlabel("In-distribution AUC (HarmBench)", fontsize=10, color=TEXT_M)
    ax.set_ylabel("Avg. OOD jailbreak detection AUC", fontsize=10, color=TEXT_M)
    save(fig, outdir, "fig_id_vs_transfer", png)


# make_presentation_figs.py latency benchmark - name: (ms, transfer AUC, class)
# classes: head = white-box (internal access: activations, or gradients for
# GradSafe); base = black-box (prompting / external models)
COST = [
    ("PersonaJB Direction",        0.0042,  0.760, "ours"),   # see DET; one
    # dot product + threshold, i.e. the cost the "Activation Probe" arm of the
    # same batch=1 benchmark measures (0.0042 ms). Cheaper than the classifier
    # it replaces (0.067 ms), which projected onto 229 traits first.
    ("Activation Probe",           0.0042,  0.71, "head"),
    # same one-dot-product cost as the Direction; only the vector differs
    ("Jailbreak Direction",              0.0042,  0.693, "head"),
    ("Refusal Direction",                0.0042,  0.670, "head"),
    ("JBShield",                   0.071,   0.69, "head"),
    # 0.242 ms was measured for the PCA(3) head; the k=229 head times 1.068x that
    # on the same sklearn pipeline (batch-1, local CPU, 3 runs), so 0.258.
    ("PCA",                        0.258,   0.721, "head"),
    ("Mahalanobis",                2.54,    0.71, "head"),
    ("JLT (SVM)",                  2.64,    0.70, "head"),
    ("JLT (RF)",                   31.4,    0.64, "head"),
    ("Perplexity",                 33.0,    0.51, "base"),
    ("Llama Guard",                43.0,    0.61, "base"),
    ("Verbalized (no CoT)",        3094.0,  0.55, "base"),
    ("Verbalized (CoT)",           3748.0,  0.52, "base"),
    ("GradSafe",                   20440.0, 0.62, "head"),
]
COST_COLOR = {"ours": C_OURS, "head": C_HEAD, "base": C_BASE}
# 95% CIs from tab:detection (Llama), matched by method name; same
# avg-transfer quantity as the y axis.
COST_CI = {
    "PersonaJB Direction": 0.004, "Activation Probe": 0.010,
    "Jailbreak Direction": 0.003, "Refusal Direction": 0.023,
    "JBShield": 0.003, "PCA": 0.008, "Mahalanobis": 0.006,
    "JLT (SVM)": 0.012, "JLT (RF)": 0.014, "Perplexity": 0.034,
    "Llama Guard": 0.027, "Verbalized (no CoT)": 0.013, "Verbalized (CoT)": 0.008,
    "GradSafe": 0.027,
}


def fig_accuracy_vs_cost(outdir, png):
    # sized for side-by-side subfigure display with the Gemma panel
    fig, ax = new_ax((4.6, 3.0))
    ax.grid(True, color=GRID, linewidth=0.8)
    offs = {
        "PersonaJB Direction": (8, -2, "left"),
        "Activation Probe": (0, 9, "center"),
        "Jailbreak Direction": (8, 2, "left"), "Refusal Direction": (8, -6, "left"),
        "JBShield": (-6, -7, "right"), "PCA": (2, -13, "center"),
        "Mahalanobis": (-3, 8, "right"), "JLT (SVM)": (7, -2, "left"),
        "JLT (RF)": (-6, 6, "right"), "Perplexity": (2, -13, "center"),
        "Llama Guard": (7, 3, "left"), "Verbalized (no CoT)": (-5, 8, "right"),
        "Verbalized (CoT)": (0, -13, "center"), "GradSafe": (-4, 8, "right"),
    }
    for name, ms, auc, cls in COST:
        c = COST_COLOR[cls]
        ci = COST_CI[name]
        ax.plot([ms, ms], [auc - ci, auc + ci], color=c, lw=1.1, alpha=0.8,
                solid_capstyle="round", zorder=3)
        if cls == "ours":
            ax.scatter(ms, auc, s=170, marker="*", color=c, zorder=5,
                       edgecolors="white", linewidths=0.7)
        else:
            ax.scatter(ms, auc, s=34, color=c,
                       zorder=4, edgecolors="white", linewidths=0.5)
        dx, dy, ha = offs[name]
        t = ax.annotate(name, (ms, auc), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7.5,
                        color=c if cls == "ours" else TEXT_M,
                        fontweight="bold" if cls == "ours" else "normal")
        halo(t)
    ax.set_xscale("log")
    ax.set_xlim(1.8e-3, 1.2e6)   # shared with the Gemma panel
    ax.set_ylim(0.465, 0.79)
    ax.set_xticks([1e-2, 1e0, 1e2, 1e4, 1e6])
    ax.set_xlabel("Detection overhead per request (ms)",
                  fontsize=10, color=TEXT_M)
    ax.set_ylabel("Avg. OOD jailbreak detection AUC", fontsize=10, color=TEXT_M)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", ls="", markerfacecolor=C_OURS,
               markeredgecolor="white", markersize=12,
               label="PersonaJB (ours)"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=C_HEAD,
               markeredgecolor="white", markersize=7,
               label="white-box baselines"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=C_BASE,
               markeredgecolor="white", markersize=7,
               label="black-box baselines"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.5,
                    framealpha=0.95, edgecolor=SPINE, borderpad=0.45)
    leg.get_frame().set_facecolor("white")
    save(fig, outdir, "fig_accuracy_vs_cost", png)



# ── Gemma-4-31B cost figure ────────────────────────────────────────────────
# AUC (+95% seed CI) = DET_GEMMA values (five families, see the note there).
# Latencies measured 2026-08-21:
# heads on the same machine as the Llama tab:timing heads (batch-1 deployable
# numpy / real estimator classes, real Gemma layer-30 + rtv tensors);
# ppl/self-exam on Gemma-4-31B and Llama Guard on an RCP GPU node
# (time-gemma-baselines job, n=48, CUDA-synced batch-1); GradSafe = official
# implementation (full backward + full cosine sweep), 2-GPU sharded, 15
# per-prompt deltas. Our numerically-equivalent fast scorer: 0.55 s/prompt.
# GradSafe AUC is the WJB-calibrated balanced-slice arm (not comparable) -> faded.
COST_GEMMA = [
    ("PersonaJB Direction", 0.0043, 0.742, 0.003, "ours", False),  # see DET_GEMMA
    ("Activation Probe", 0.0043,  0.651, 0.006, "head", False),
    ("Jailbreak Direction",    0.0043,  0.677, 0.001, "head", False),
    ("Refusal Direction",      0.0043,  0.662, 0.024, "head", False),
    ("JBShield",         0.0425,  0.674, 0.002, "head", False),
    ("PCA",              0.1155,  0.694, 0.003, "head", False),  # k=240; 0.1077 x 1.072, see COST
    ("Mahalanobis",      1.4653,  0.696, 0.001, "head", False),
    ("JLT (SVM)",        1.4926,  0.675, 0.002, "head", False),
    ("JLT (RF)",         7.8416,  0.687, 0.004, "head", False),
    ("Llama Guard",      33.6,    0.566, 0.034, "base", False),
    ("Perplexity",       86.5,    0.505, 0.029, "base", False),
    ("Verbalized (no CoT)", 5052.0, 0.516, 0.014, "base", False),
    ("Verbalized (CoT)", 13507.0, 0.522, 0.009, "base", False),
    ("GradSafe",         173400.0, 0.702, 0.028, "head", False),  # see GEMMA_BALANCED
]


def fig_accuracy_vs_cost_gemma(outdir, png):
    fig, ax = new_ax((4.6, 3.0))
    ax.grid(True, color=GRID, linewidth=0.8)
    offs = {
        "PersonaJB Direction": (8, -2, "left"),
        "Activation Probe": (2, 9, "center"),
        "Jailbreak Direction": (8, 3, "left"), "Refusal Direction": (8, -5, "left"),
        "JBShield": (5, -12, "left"), "PCA": (0, -13, "center"),
        "Mahalanobis": (-5, 7, "right"), "JLT (SVM)": (6, -10, "left"),
        "JLT (RF)": (0, 8, "center"), "Perplexity": (2, -13, "center"),
        "Llama Guard": (-2, 8, "right"),
        "Verbalized (no CoT)": (-5, 8, "right"),
        "Verbalized (CoT)": (0, -13, "center"), "GradSafe": (-6, 5, "right"),
    }
    for name, ms, auc, ci, cls, faded in COST_GEMMA:
        c = COST_COLOR[cls]
        a = 0.45 if faded else 1.0
        ax.plot([ms, ms], [auc - ci, auc + ci], color=c, lw=1.1, alpha=0.8 * a,
                solid_capstyle="round", zorder=3)
        if cls == "ours":
            ax.scatter(ms, auc, s=170, marker="*", color=c, zorder=5,
                       edgecolors="white", linewidths=0.7)
        else:
            ax.scatter(ms, auc, s=34, color=c, alpha=a,
                       zorder=4, edgecolors="white", linewidths=0.5)
        dx, dy, ha = offs[name]
        t = ax.annotate(name, (ms, auc), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7.5, alpha=max(a, 0.7),
                        color=c if cls == "ours" else TEXT_M,
                        fontweight="bold" if cls == "ours" else "normal")
        halo(t)
    ax.set_xscale("log")
    ax.set_xlim(1.8e-3, 1.2e6)   # shared with the Llama panel
    ax.set_ylim(0.465, 0.79)
    ax.set_xticks([1e-2, 1e0, 1e2, 1e4, 1e6])
    ax.set_xlabel("Detection overhead per request (ms)",
                  fontsize=10, color=TEXT_M)
    save(fig, outdir, "fig_accuracy_vs_cost_gemma", png)



# Side-by-side rendering gives each panel only 2.7in of the 5.5in textwidth, so
# the paired variants use short labels and a smaller canvas (less downscaling).
COST_SHORT = {
    "PersonaJB Direction": "PersonaJB\nDirection",
    "Activation Probe": "Activation\nProbe",
    "JBShield": "JBShield", "PCA": "PCA", "Mahalanobis": "Mahalanobis",
    "JLT (SVM)": "JLT-SVM", "JLT (RF)": "JLT-RF", "Perplexity": "Perplexity",
    "Jailbreak Direction": "Jailbreak Direction",
    "Refusal Direction": "Refusal Direction",
    "Llama Guard": "Llama Guard", "Verbalized (no CoT)": "Verbalized (no CoT)",
    "Verbalized (CoT)": "Verbalized (CoT)", "GradSafe": "GradSafe",
}


# Their markers land inside the one-dot-product column (four methods, same x),
# so these two are labelled out in free space with a hairline leader.
LEADERS = {"Jailbreak Direction", "Refusal Direction"}


def fig_cost_pair(outdir, png):
    r"""Both models on shared axes, sized for 0.49\linewidth subfigures."""
    for tag, rows, ylab, leg_on in (
            ("llama", [(n, ms, auc, COST_CI[n], cls, False) for n, ms, auc, cls in COST],
             True, True),
            ("gemma", COST_GEMMA, False, False)):
        fig, ax = new_ax((3.7, 2.8))
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(labelsize=8)
        # dense log minor ticks smear into a band once the panel is downscaled
        # to 0.49\linewidth, so drop them
        ax.tick_params(which='minor', length=0)
        offs = OFFS_PAIR[tag]
        for name, ms, auc, ci, cls, faded in rows:
            c = COST_COLOR[cls]
            a = 0.22 if faded else 1.0   # faded = not directly comparable
            ax.plot([ms, ms], [auc - ci, auc + ci], color=c, lw=1.0,
                    alpha=0.8 * a, solid_capstyle="round", zorder=3)
            if cls == "ours":
                ax.scatter(ms, auc, s=150, marker="*", color=c, zorder=5,
                           edgecolors="white", linewidths=0.6)
            else:
                ax.scatter(ms, auc, s=26, color=c, alpha=a, zorder=4,
                           edgecolors="white", linewidths=0.4)
            dx, dy, ha = offs[name]
            lead = dict(arrowstyle="-", lw=0.45, color=TEXT_M, alpha=0.65,
                        shrinkA=1.0, shrinkB=2.5) if name in LEADERS else None
            t = ax.annotate(COST_SHORT[name], (ms, auc),
                            textcoords="offset points", xytext=(dx, dy), ha=ha,
                            fontsize=6.8, alpha=0.45 if faded else 1.0,
                            color=c if cls == "ours" else TEXT_M,
                            fontweight="bold" if cls == "ours" else "normal",
                            arrowprops=lead)
            halo(t, 2.2)
        ax.set_xscale("log")
        # shared range, trimmed to just past the slowest method (Gemma GradSafe,
        # 1.7e5 ms) so neither panel carries dead space on the right
        ax.set_xlim(2.2e-3, 3.0e5)
        # the two panels no longer share a y range: Llama needs 0.794 for the
        # Direction's whisker, Gemma's tallest element is 0.756
        ax.set_ylim(0.465, 0.79 if tag == "llama" else 0.772)
        ax.set_xticks([1e-2, 1e0, 1e2, 1e4])
        ax.set_xlabel("Detection overhead / request (ms)", fontsize=9,
                      color=TEXT_M)
        if ylab:
            ax.set_ylabel("Avg. OOD jailbreak detection AUC", fontsize=9,
                          color=TEXT_M)
        if leg_on:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], marker="*", ls="", markerfacecolor=C_OURS,
                       markeredgecolor="white", markersize=10, label="ours"),
                Line2D([0], [0], marker="o", ls="", markerfacecolor=C_HEAD,
                       markeredgecolor="white", markersize=6, label="white-box"),
                Line2D([0], [0], marker="o", ls="", markerfacecolor=C_BASE,
                       markeredgecolor="white", markersize=6, label="black-box"),
            ]
            leg = ax.legend(handles=handles, loc="lower left", fontsize=6.8,
                            framealpha=0.95, edgecolor=SPINE, borderpad=0.4,
                            handletextpad=0.35, labelspacing=0.3)
            leg.get_frame().set_facecolor("white")
        save(fig, outdir, f"fig_cost_pair_{tag}", png)


def fig_cost_pair_combined(outdir, png):
    r"""Both models side by side on shared x/y axes as one full-width
    figure: one y label, one x label, no subcaptions, panels flush. Sized
    at \linewidth (5.5in) for a top float in the detection section (replaces
    the two fig_cost_pair PDFs and the earlier stacked wrapfigure form)."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.80), dpi=200,
                             sharex=True, sharey=True)
    panels = (
        ("llama", [(n, ms, auc, COST_CI[n], cls, False)
                   for n, ms, auc, cls in COST], "Llama-3.1-8B", "#A6403A", True),
        ("gemma", COST_GEMMA, "Gemma-4-31B", "#C88A2D", False),
    )
    for ax, (tag, rows, model, mcol, leg_on) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(SPINE)
            s.set_linewidth(0.9)
        ax.tick_params(colors=TEXT_M, labelsize=8, length=0)
        ax.tick_params(which='minor', length=0)
        ax.grid(True, color=GRID, linewidth=0.8)
        offs = OFFS_WIDE[tag]
        for name, ms, auc, ci, cls, faded in rows:
            c = COST_COLOR[cls]
            a = 0.22 if faded else 1.0
            ax.plot([ms, ms], [auc - ci, auc + ci], color=c, lw=1.0,
                    alpha=0.8 * a, solid_capstyle="round", zorder=3)
            if cls == "ours":
                ax.scatter(ms, auc, s=150, marker="*", color=c, zorder=5,
                           edgecolors="white", linewidths=0.6)
            else:
                ax.scatter(ms, auc, s=26, color=c, alpha=a, zorder=4,
                           edgecolors="white", linewidths=0.4)
            dx, dy, ha = offs[name]
            is_lead = name in LEADERS_WIDE
            lead = dict(arrowstyle="-", lw=0.45, color=TEXT_M, alpha=0.65,
                        shrinkA=1.0, shrinkB=2.5) if is_lead else None
            label = COST_SHORT[name].replace("\n", " ") if is_lead else COST_SHORT[name]
            t = ax.annotate(label, (ms, auc),
                            textcoords="offset points", xytext=(dx, dy), ha=ha,
                            fontsize=6.8, alpha=0.45 if faded else 1.0,
                            color=c if cls == "ours" else TEXT_M,
                            fontweight="bold" if cls == "ours" else "normal",
                            arrowprops=lead)
            halo(t, 2.2)
        ax.set_xscale("log")
        ax.set_xlim(2.2e-3, 3.0e5)
        ax.set_xticks([1e-2, 1e0, 1e2, 1e4])
        # model tag in-panel (replaces the removed subcaption)
        mt = ax.text(0.975, 0.95, model, transform=ax.transAxes, ha="right",
                     va="top", fontsize=7.5, fontweight="bold", color=mcol)
        halo(mt, 2.4)
        if leg_on:
            handles = [
                Line2D([0], [0], marker="*", ls="", markerfacecolor=C_OURS,
                       markeredgecolor="white", markersize=10, label="ours"),
                Line2D([0], [0], marker="o", ls="", markerfacecolor=C_HEAD,
                       markeredgecolor="white", markersize=6, label="white-box"),
                Line2D([0], [0], marker="o", ls="", markerfacecolor=C_BASE,
                       markeredgecolor="white", markersize=6, label="black-box"),
            ]
            # one boxed row above the panels, so the lower-left corner is
            # free for the leader labels of the one-dot-product column
            leg = fig.legend(handles=handles, loc="lower center", ncol=3,
                             bbox_to_anchor=(0.54, 0.975), fontsize=7.0,
                             framealpha=0.95, edgecolor=SPINE, borderpad=0.35,
                             handletextpad=0.2, columnspacing=0.9)
            leg.get_frame().set_facecolor("white")
    # one shared y range (Llama's 0.79 top clears the Direction whisker)
    axes[0].set_ylim(0.465, 0.79)
    axes[0].set_ylabel("Avg. OOD jailbreak\ndetection AUC", fontsize=8.5,
                       color=TEXT_M)
    fig.supxlabel("Detection overhead / request (ms)", fontsize=8.5,
                  color=TEXT_M, x=0.54, y=0.0)
    # flush panels (shared y hides the right panel's tick labels)
    fig.subplots_adjust(wspace=0.05, left=0.09, right=0.99, top=0.975,
                        bottom=0.20)
    fig.savefig(outdir / "fig_cost_pair_combined.pdf", bbox_inches="tight")
    if png:
        fig.savefig(outdir / "fig_cost_pair_combined.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", outdir / "fig_cost_pair_combined")


OFFS_PAIR = {
    "llama": {
        "PersonaJB Direction": (8, -2, "left"), "Activation Probe": (8, 4, "left"),
        "Jailbreak Direction": (2, -34, "left"),
        "Refusal Direction": (2, -50, "left"),
        # JBShield sits 1.2 decades right of that column, so its label goes
        # up-right instead of left into the column
        "JBShield": (4, 7, "left"), "PCA": (5, -4, "left"),
        "Mahalanobis": (4, 7, "left"), "JLT (SVM)": (6, -3, "left"),
        "JLT (RF)": (-5, 5, "right"), "Perplexity": (0, -11, "center"),
        "Llama Guard": (6, 1, "left"), "Verbalized (no CoT)": (-4, 7, "right"),
        "Verbalized (CoT)": (0, -11, "center"), "GradSafe": (-3, 7, "right"),
    },
    "gemma": {
        "PersonaJB Direction": (8, -2, "left"), "Activation Probe": (8, -6, "left"),
        "Jailbreak Direction": (2, -40, "left"),
        "Refusal Direction": (2, -54, "left"),
        "JBShield": (4, 7, "left"), "PCA": (0, 8, "center"),
        "Mahalanobis": (5, 6, "left"), "JLT (SVM)": (5, -8, "left"),
        # JLT-RF's five-family AUC (0.687) sits close enough to Mahalanobis
        # (0.696) that a label above the marker runs into it -> place it right.
        "JLT (RF)": (6, -3, "left"), "Perplexity": (0, -11, "center"),
        "Llama Guard": (-3, 6, "right"), "Verbalized (no CoT)": (-4, 7, "right"),
        "Verbalized (CoT)": (-14, -11, "center"), "GradSafe": (-4, 5, "right"),
    },
}

# fig_cost_pair_combined (side-by-side, ~2.45in per panel): same as OFFS_PAIR
# except where the narrower panel made a right-hand label run into a neighbour.
OFFS_WIDE = {tag: dict(d) for tag, d in OFFS_PAIR.items()}
OFFS_WIDE["llama"].update({
    "JBShield": (0, -11, "center"),          # up-right ran into the PCA marker
    "Verbalized (no CoT)": (0, 8, "center"), # left-hand text met Refusal Direction's
    # 2.25in-tall panels: 0.01 AUC is ~4pt, so the leader labels sit higher to
    # clear the legend, and Activation Probe drops clear of the PersonaJB label
    "Activation Probe": (8, 1, "left"),
    "Jailbreak Direction": (2, -30, "left"),
    "Refusal Direction": (2, -40, "left"),
})
OFFS_WIDE["gemma"].update({
    "JBShield": (5, -3, "left"),             # up-right ran into the PCA marker
    "Jailbreak Direction": (2, -36, "left"),
    "Refusal Direction": (2, -46, "left"),
    "Llama Guard": (6, 2, "left"),           # left-hand text met Jailbreak Direction's
})
# 1.8in-tall panels: the one-dot-product column (Activation Probe, Jailbreak and
# Refusal Direction) stacks below its markers on single-line leader labels
LEADERS_WIDE = LEADERS | {"Activation Probe"}
OFFS_WIDE["llama"].update({
    "Activation Probe": (2, -30, "left"),     # clears JBShield's label
    "Jailbreak Direction": (2, -40, "left"),
    "Refusal Direction": (2, -50, "left"),
})
OFFS_WIDE["gemma"].update({
    "Jailbreak Direction": (2, -24, "left"),
    "Refusal Direction": (2, -35, "left"),
    "Activation Probe": (2, -46, "left"),
    "Verbalized (no CoT)": (0, 8, "center"),  # right-aligned text met Refusal Direction's
})


def fig_layer_sweep(outdir, png, csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    layers = [int(r["layer"]) for r in rows]
    indom = [float(r["best_test_human_test_auc"]) for r in rows]
    fams = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
    transfer = [np.mean([float(r[f"best_transfer_{f}"]) for f in fams])
                for r in rows]
    fig, ax = new_ax((5.6, 2.9))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.axvline(16, color="#9A9A9A", lw=1.2, ls=(0, (1, 2)), zorder=1)
    ax.plot(layers, indom, color=C_HEAD, lw=1.8, marker="o", ms=3.2,
            markerfacecolor=C_HEAD, markeredgecolor="white",
            markeredgewidth=0.4, label="HarmBench (in-domain test)", zorder=3)
    ax.plot(layers, transfer, color=C_RAW, lw=1.8, marker="s", ms=3.2,
            markerfacecolor=C_RAW, markeredgecolor="white",
            markeredgewidth=0.4, label="mean transfer (5 unseen families)",
            zorder=3)
    t = ax.text(16, 1.01, "layer 16", color=TEXT_M, fontsize=8.5, ha="center",
                va="bottom", transform=ax.get_xaxis_transform())
    halo(t)
    ax.set_xlim(-0.6, 31.6)
    ax.set_xticks(range(0, 32, 4))
    ax.set_xlabel("Layer", fontsize=10, color=TEXT_M)
    ax.set_ylabel("Jailbreak Detection AUC", fontsize=10, color=TEXT_M)
    leg = ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95,
                    edgecolor=SPINE, borderpad=0.5)
    leg.get_frame().set_facecolor("white")
    save(fig, outdir, "fig_layer_sweep_auc", png)


# tab:attack, HarmBench originally-refused column
ATK = [
    ("Steer-$w$ (raw control)", 0.76,  "raw"),
    ("Steer-$w$ (persona)", 0.747, "ours"),
    ("Steer-traits",             0.666, "ours"),
    ("GPTFuzz",             0.59,  "base"),
    ("GCG",                 0.44,  "base"),
    ("PAIR",                0.39,  "base"),
    ("PAP",                 0.28,  "base"),
    ("PEZ",                 0.15,  "base"),
]


def fig_attack_refused(outdir, png):
    fig, ax = new_ax((5.8, 3.2))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # The larger gap makes the methodological comparison legible without a
    # plot-filling legend: names identify the methods and color remains a
    # redundant cue (trait steering, raw-activation control, standard attack).
    ys = np.array([7.8, 6.8, 5.8, 4.2, 3.2, 2.2, 1.2, 0.2])
    for y, (name, v, cls) in zip(ys, ATK):
        pct = v * 100
        value_label = (f"{pct:.0f}%" if np.isclose(pct, round(pct))
                       else f"{pct:.1f}%")
        ax.barh(y, pct, height=0.60, color=CLASS_COLOR[cls], zorder=3)
        ax.text(pct + 0.9, y, value_label, va="center", ha="left",
                fontsize=8.5, color=TEXT_M,
                fontweight="bold" if cls == "ours" else "normal")

    # The best standard attack is the natural reference for the steering bars.
    # Keep the line subtle so it supports the comparison rather than becoming a
    # second graphical element competing with the bars.
    best_standard = max(v for _, v, cls in ATK if cls == "base") * 100
    ax.axvline(best_standard, color="#7D8790", linewidth=0.9,
               linestyle=(0, (2.5, 2.5)), zorder=2)
    ax.text(best_standard + 0.8, 8.40, "best standard · 59%",
            ha="left", va="center", fontsize=7.4, color="#68717A")
    ax.text(0, 8.40, "STEERING INTERVENTIONS", ha="left", va="center",
            fontsize=7.4, color=TEXT_M, fontweight="bold")
    ax.text(0, 4.92, "STANDARD ATTACKS", ha="left", va="center",
            fontsize=7.4, color=TEXT_M, fontweight="bold")

    ax.set_yticks(ys)
    tick_labels = ax.set_yticklabels([a[0] for a in ATK], fontsize=8.8)
    for label, (_, _, cls) in zip(tick_labels, ATK):
        label.set_fontweight("bold" if cls == "ours" else "normal")
    ax.set_ylim(-0.35, 8.72)
    ax.set_xlim(0, 84)
    ax.set_xticks(range(0, 81, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 81, 20)])
    ax.set_xlabel("Attack success rate on originally-refused HarmBench prompts",
                  fontsize=9.5, color=TEXT_M)
    save(fig, outdir, "fig_attack_refused", png)


def fig_readout_probe(outdir, png, npz_path):
    z = np.load(npz_path)
    aucs, stds, W = z["aucs"], z["stds"], int(z["W"])
    ci = 1.96 * stds / np.sqrt(50)          # 50-seed mean, per the recast run
    x = np.arange(-W, 1)
    fig, ax = new_ax((4.7, 3.0))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.axvspan(-4.5, 0.5, color="0.93", zorder=0)
    ax.axhline(0.5, ls=(0, (6, 4)), color="#8A8A8A", lw=1.1, zorder=1)
    keep = ~np.isnan(aucs)
    ax.plot(x[keep], aucs[keep], color=C_HEAD, lw=1.8, marker="o", ms=3.2,
            markerfacecolor=C_HEAD, markeredgecolor="white",
            markeredgewidth=0.4, zorder=3)
    ax.fill_between(x[keep], (aucs - ci)[keep], (aucs + ci)[keep],
                    color=C_HEAD, alpha=0.18, linewidth=0, zorder=2)
    ax.plot([0], [aucs[-1]], marker="*", ms=12, color=C_HEAD,
            markeredgecolor="white", markeredgewidth=0.7, zorder=5)
    t = ax.text(0.985, 0.504, "chance", color=TEXT_M, fontsize=8.5,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
    halo(t)
    ax.set_ylim(0.44, 1.0)
    t = ax.text(-2.0, 0.972, "assistant\ntemplate", ha="center", va="top",
                fontsize=8.5, color=TEXT_M)
    halo(t)
    ax.set_xticks([-25, -20, -15, -10, -5, 0])
    ax.set_xlabel("token position relative to the readout token",
                  fontsize=10, color=TEXT_M)
    ax.set_ylabel("held-out detection AUC\n(fresh probe per position)",
                  fontsize=10, color=TEXT_M)
    save(fig, outdir, "fig_persona_readout_probe", png)


# fig_wtrait_personas: signed cosine of each persona direction with the trait
# classifier normal (top 15 by |cos|), values from the deployed figure
WTRAIT = [
    ("relativist", -0.197), ("constructivist", -0.155),
    ("anthropocentric", -0.152), ("eclectic", -0.138), ("dominant", -0.138),
    ("convergent", -0.132), ("pacifist", -0.125), ("sardonic", 0.122),
    ("impatient", -0.121), ("urgent", -0.116), ("reductionist", -0.113),
    ("wry", 0.112), ("impulsive", 0.107), ("humble", -0.105),
    ("creative", -0.105),
]
C_JAILBRK = "#D28181"
C_REFUSED = "#6A90BA"


def fig_wtrait_personas(outdir, png):
    fig, ax = new_ax((5.2, 3.6))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    chance = 1 / np.sqrt(4096)
    ys = np.arange(len(WTRAIT))[::-1]
    for y, (name, v) in zip(ys, WTRAIT):
        c = C_JAILBRK if v > 0 else C_REFUSED
        ax.barh(y, abs(v), height=0.62, color=c, zorder=3)
        ax.text(abs(v) + 0.003, y, f"{v:+.3f}", va="center", ha="left",
                fontsize=8, color=TEXT_M)
    ax.set_yticks(ys)
    ax.set_yticklabels([w[0] for w in WTRAIT], fontsize=9)
    ax.axvline(chance, ls=(0, (6, 4)), color="#8A8A8A", lw=1.1, zorder=1)
    t = ax.text(chance, -0.9, "chance\n$1/\\sqrt{4096}$", ha="center",
                va="top", fontsize=8, color=TEXT_M)
    halo(t)
    ax.set_xlim(0, 0.26)
    ax.set_ylim(-2.4, len(WTRAIT) - 0.4)
    ax.set_xlabel("$|\\cos|$ with the trait classifier normal $w_{\\mathrm{trait}}$",
                  fontsize=10, color=TEXT_M)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C_JAILBRK, C_REFUSED)]
    leg = ax.legend(handles, ["toward jailbreak ($+w$)",
                              "toward refusal ($-w$)"],
                    loc="lower right", fontsize=8.5, framealpha=0.95,
                    edgecolor=SPINE, borderpad=0.5)
    leg.get_frame().set_facecolor("white")
    save(fig, outdir, "fig_wtrait_personas", png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str,
        default="full_trait_output/plots/paper_figs")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--layer_csv", type=str,
        default="full_trait_output/all_layers_sweep_summary.csv")
    parser.add_argument("--probe_npz", type=str,
        default="full_trait_output/per_position_probe_recast_auc.npz")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_detection_transfer(outdir, args.png)
    fig_detection_transfer_transposed(outdir, args.png)
    fig_id_vs_transfer(outdir, args.png)
    fig_accuracy_vs_cost(outdir, args.png)
    fig_accuracy_vs_cost_gemma(outdir, args.png)
    fig_cost_pair(outdir, args.png)
    fig_cost_pair_combined(outdir, args.png)
    fig_layer_sweep(outdir, args.png, args.layer_csv)
    fig_attack_refused(outdir, args.png)
    fig_readout_probe(outdir, args.png, args.probe_npz)
    fig_wtrait_personas(outdir, args.png)


if __name__ == "__main__":
    main()
