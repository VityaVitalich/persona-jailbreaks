#!/usr/bin/env python3
"""
Appendix control figures in the style of fig_direction_story (emitted at the paper's \\linewidth, 397pt).
  fig_controls_share       left: share of the refusal and jailbreak directions' squared norm in each 229-dim space;
                           right: share in the trait span of the real directions and of their shuffled-label nulls
  fig_controls_directions  folded AUC on HarmBench (test set, 50 splits) and transfer (mean of 5 families) of each
                           direction: full, its persona component only, persona component removed, random span removed
  fig_controls_spaces      folded AUC of the PersonaJB detector on each 229-direction dictionary: full, persona
                           component removed, random span removed (draws: dot = mean, whisker = min-max); below them
                           the refusal and jailbreak directions used as single scores h.v (appendix_controls_data.json)
Data: appendix_controls_data.json (appendix_controls_data.py), remove_persona_all_controls.json, persona_row_check.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "full_trait_tools"))
from plot_paper_result_figs import GRID, SPINE, TEXT_M, halo  # noqa: E402  (also sets the paper rcParams)
D = json.load(open(HERE / "appendix_controls_data.json"))
R = json.load(open(HERE / "remove_persona_all_controls.json")); P = json.load(open(HERE / "persona_row_check.json"))
W = 397 / 72
C_REF, C_DMU = "#1D5B8C", "#C88A2D"
C_REF_L, C_DMU_L = "#8FB0CC", "#E3C48F"
C_GRAY, C_GRAY_L, C_PER = "#7A7A7A", "#C2C2C2", "#A6403A"

def style(ax):
    ax.set_facecolor("white"); ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.7)
    for s in ax.spines.values(): s.set_color(SPINE); s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_M, labelsize=6.8, length=0, pad=2)

def save(fig, name):
    for e in ("pdf", "png"): fig.savefig(HERE / f"figs/{name}.{e}", dpi=200)

# ---------- fig_controls_share ----------
SPACES = [("persona", "Trait vectors"), ("alpaca_pca", "Alpaca PCs"), ("alpaca_acts", "Alpaca activations"),
          ("norobots_acts", "No Robots activations"), ("wiki_acts", "Wikipedia activations"), ("random", "Random directions")]
DIRS = [("refusal", "Refusal direction", C_REF), ("strat_refusal", "Shuffled-label refusal", C_REF_L),
        ("dmu", "Jailbreak direction", C_DMU), ("strat_dmu", "Shuffled-label jailbreak", C_DMU_L),
        ("shuf_alpaca", "Shuffled Alpaca contrast", C_GRAY_L), ("random_direction", "Random direction", "#E0E0E0")]
fig, (aL, aR) = plt.subplots(1, 2, figsize=(W, 2.3), dpi=200, gridspec_kw={"wspace": 1.0})
for ax in (aL, aR): style(ax)
yL = -np.arange(len(SPACES)); h = 0.36
for off, tn, c in ((h / 2, "refusal", C_REF), (-h / 2, "dmu", C_DMU)):
    for y, (sn, _) in zip(yL, SPACES):
        s = D["share"][tn][sn]; m = 100 * s["mean"]
        aL.barh(y + off, m, height=h, color=c, alpha=0.88, zorder=3)
        if s["max"] > s["min"]: aL.plot([100 * s["min"], 100 * s["max"]], [y + off] * 2, color="#333333", lw=0.7, zorder=4)
        t = aL.text(100 * s["max"] + 1.2, y + off, f"{m:.0f}", va="center", fontsize=5.8, color=c); halo(t, 1.8)
aL.set_yticks(yL); aL.set_yticklabels([s[1] for s in SPACES], fontsize=6.8, color=TEXT_M)
aL.set_xlim(0, 60); aL.set_xlabel("share of norm (%)", fontsize=7, color=TEXT_M, labelpad=2)
aL.set_title("refusal and jailbreak directions,\nshare in each 229-dim space", fontsize=7, color=TEXT_M, pad=3)
aL.legend(handles=[Patch(color=C_REF, label="refusal direction"), Patch(color=C_DMU, label="jailbreak direction")],
          loc="lower right", fontsize=6, frameon=False, handlelength=1.0, borderaxespad=0.2)
yR = -np.arange(len(DIRS))
for y, (dn, lab, c) in zip(yR, DIRS):
    s = D["share_persona"][dn]; m = 100 * s["mean"]
    aR.barh(y, m, height=0.6, color=c, zorder=3)
    if s["p95"] > s["p5"]: aR.plot([100 * s["p5"], 100 * s["p95"]], [y, y], color="#333333", lw=0.7, zorder=4)
    t = aR.text(100 * s["p95"] + 1.2, y, f"{m:.0f}", va="center", fontsize=5.8, color=TEXT_M); halo(t, 1.8)
aR.set_yticks(yR); aR.set_yticklabels([d[1] for d in DIRS], fontsize=6.8, color=TEXT_M)
aR.set_xlim(0, 60); aR.set_xlabel("share of norm (%)", fontsize=7, color=TEXT_M, labelpad=2)
aR.set_title("real and shuffled-label directions,\nshare in persona space", fontsize=7, color=TEXT_M, pad=3)
fig.subplots_adjust(left=0.215, right=0.975, top=0.85, bottom=0.16)
save(fig, "fig_controls_share")

# ---------- fig_controls_directions ----------
ROWS = [("refusal", "Refusal direction", C_REF), ("strat_refusal", "Shuffled-label refusal", C_REF_L),
        ("dmu", "Jailbreak direction", C_DMU), ("strat_dmu", "Shuffled-label jailbreak", C_DMU_L),
        ("shuf_alpaca", "Shuffled Alpaca contrast", C_GRAY), ("random_direction", "Random direction", C_GRAY_L)]
VAR = [("full", "full direction", "o", "white"), ("persona_component", "persona component only", "*", None),
       ("persona_removed", "persona component removed", "X", C_PER), ("random_removed", "random span removed", "D", "#9AA4AD")]
fig, axs = plt.subplots(1, 2, figsize=(W, 2.35), dpi=200, sharey=True, gridspec_kw={"wspace": 0.08})
ys = -np.arange(len(ROWS))
for ax, key, title, xl in ((axs[0], "hb", "HarmBench (test set)", (0.60, 0.90)), (axs[1], "transfer", "Avg. transfer", (0.58, 0.72))):
    style(ax); ax.set_xticks(np.round(np.arange(xl[0] + (0.05 if key == "hb" else 0.02), xl[1] - 0.001, 0.05 if key == "hb" else 0.02), 2))
    for y, (dn, lab, c) in zip(ys, ROWS):
        v = D["auc"][dn]; xs = [v[k][key] for k, *_ in VAR]
        ax.plot([min(xs), max(xs)], [y, y], color=c, lw=1.0, alpha=0.5, zorder=2)
        if key == "hb":
            f = v["full"]; ax.plot([f["hb"] - f["hb_ci"], f["hb"] + f["hb_ci"]], [y, y], color=c, lw=4.5, alpha=0.22, solid_capstyle="round", zorder=1)
        for (k, _, mk, fc), x in zip(VAR, xs):
            ax.scatter([x], [y], marker=mk, s={"o": 30, "*": 42, "X": 20, "D": 11}[mk], zorder={"o": 6, "*": 5, "X": 4, "D": 3}[mk],
                       facecolor=(c if fc is None else fc), edgecolor=(c if mk == "o" else "white"), lw=1.1 if mk == "o" else 0.4)
    ax.set_xlim(*xl); ax.set_title(title, fontsize=7.2, color=TEXT_M, pad=3)
    ax.set_xlabel("jailbreak detection AUC", fontsize=7, color=TEXT_M, labelpad=2)
axs[0].set_yticks(ys); axs[0].set_yticklabels([r[1] for r in ROWS], fontsize=6.8, color=TEXT_M)
hd = [Line2D([], [], ls="", marker=mk, markersize=6 if mk == "*" else 4.5, markerfacecolor=("#666666" if fc is None else fc),
             markeredgecolor="#666666" if mk == "o" else "white", label=lab) for _, lab, mk, fc in VAR]
fig.legend(handles=hd, loc="lower center", ncol=4, fontsize=6.2, frameon=False, handletextpad=0.3, columnspacing=1.0)
fig.subplots_adjust(left=0.235, right=0.985, top=0.9, bottom=0.27)
save(fig, "fig_controls_directions")

# ---------- fig_controls_spaces ----------
SROWS = [("Trait vectors (PersonaJB)", None), ("Wikipedia activations", "Wikipedia acts"), ("No Robots activations", "No Robots acts"),
         ("Alpaca activations", "Alpaca acts"), ("Shuffled Alpaca contrasts", "Shuffled contrasts"),
         ("Alpaca PCs", "Alpaca PCA-229"), ("Random directions", "Random Gaussian"),
         ("Refusal direction", "dir:refusal"), ("Jailbreak direction", "dir:dmu")]
SV = [("whole", "full space or direction", "o", "white", "#1D5B8C"), ("persona", "persona component removed", "X", C_PER, C_PER),
      ("random span", "random span removed", "D", "#9AA4AD", "#9AA4AD")]
fig, axs = plt.subplots(1, 2, figsize=(W, 2.85), dpi=200, sharey=True, gridspec_kw={"wspace": 0.08})
ys = -np.arange(len(SROWS))
for ax, ki, title, xl in ((axs[0], 0, "HarmBench (test set)", (0.70, 0.90)), (axs[1], 1, "Avg. transfer", (0.62, 0.78))):
    style(ax); ax.set_xticks([0.75, 0.80, 0.85] if ki == 0 else [0.65, 0.70, 0.75])
    for y, (lab, key) in zip(ys, SROWS):
        if key is None:
            x = P["hb"] if ki == 0 else P["transfer"]
            ax.scatter([x], [y], marker="o", s=22, facecolor="white", edgecolor="#1D5B8C", lw=1.1, zorder=4); continue
        pts = []
        for k, _, mk, fc, ec in SV:
            if key.startswith("dir:"):  # single direction: appendix_controls_data.json (HarmBench test set, 50 splits)
                kk = {"whole": "full", "persona": "persona_removed", "random span": "random_removed"}[k]
                arr = np.array([[D["auc"][key[4:]][kk]["hb"], D["auc"][key[4:]][kk]["transfer"]]])[:, ki]
            else:
                arr = np.array([o[k] if k == "whole" else o[k][1:] for o in R[key]])[:, ki]
            m = arr.mean(); pts.append(m)
            if arr.max() > arr.min(): ax.plot([arr.min(), arr.max()], [y, y], color=ec, lw=0.8, alpha=0.7, zorder=3)
            ax.scatter([m], [y], marker=mk, s={"o": 30, "X": 20, "D": 11}[mk], zorder={"o": 6, "X": 5, "D": 4}[mk], facecolor=fc, edgecolor=ec if mk == "o" else "white", lw=1.1 if mk == "o" else 0.4)
        ax.plot([min(pts), max(pts)], [y, y], color="#999999", lw=0.8, alpha=0.5, zorder=2)
    ax.axhline(-0.5, color=SPINE, lw=0.6); ax.axhline(-6.5, color=SPINE, lw=0.6)
    ax.set_xlim(*xl); ax.set_title(title, fontsize=7.2, color=TEXT_M, pad=3)
    ax.set_xlabel("jailbreak detection AUC", fontsize=7, color=TEXT_M, labelpad=2)
axs[0].set_yticks(ys); axs[0].set_yticklabels([r[0] for r in SROWS], fontsize=6.8, color=TEXT_M)
hd = [Line2D([], [], ls="", marker=mk, markersize=4.5, markerfacecolor=fc, markeredgecolor=ec if mk == "o" else "white", label=lab)
      for _, lab, mk, fc, ec in SV]
fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=6.2, frameon=False, handletextpad=0.3, columnspacing=1.2)
fig.subplots_adjust(left=0.235, right=0.985, top=0.91, bottom=0.22)
save(fig, "fig_controls_spaces")
print("done")
