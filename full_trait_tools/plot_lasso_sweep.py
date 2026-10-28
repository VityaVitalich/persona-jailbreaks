#!/usr/bin/env python3
"""
plot_lasso_sweep.py -- figures for trait_lasso_sweep.py.

fig_lasso_curve   how detection quality moves as the trait budget shrinks:
                  in-distribution AUC, mean OOD transfer AUC, and how much of the
                  dense PersonaJB direction a k-trait subspace recovers. Three
                  selection rules share every budget so "a few traits are enough"
                  is separated from "any few traits are enough".
fig_lasso_traits  which traits the L1 path keeps: selection frequency over the 50
                  splits at a budget of ten, and the budget at which each trait
                  first enters the support.

Bands are +-1 standard error over the 50 pool splits. The splits are the noise
source the curves are being compared against, so the SEM is what says whether two
rules differ; the single-split spread is much wider (a 288-row test set) and is
reported in the console summary instead.
"""
from __future__ import annotations

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
from plot_paper_result_figs import GRID, SPINE, TEXT_M, halo  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "full_trait_output/plots/lasso"

C_LASSO, C_UNI, C_RAND = "#A6403A", "#1D5B8C", "#9AA4AD"
C_DENSE = "#444444"
C_POS, C_NEG = "#A6403A", "#1D5B8C"
RULES = [("lasso", "L1 selection", C_LASSO),
         ("univar", "Top-$k$ single traits", C_UNI),
         ("random", "Random $k$ traits", C_RAND)]
SHRUNK = ("shrunk", "L1 coefficients (shrunk)", C_LASSO)


def budget_point(seed, k):
    """Densest L1 path point whose support fits in a budget of k traits."""
    ok = [e for e in seed["path"] if 0 < e["k"] <= k]
    return max(ok, key=lambda e: e["k"]) if ok else None


def curves(rec):
    """mean/SEM of every (rule, metric) over seeds, on the shared k grid."""
    ks, fams, seeds = rec["kgrid"], rec["families"], rec["seeds"]
    per = {r: {m: [] for m in ("hb", "ood", "cos")} for r, _, _ in RULES + [SHRUNK]}
    for k in ks:
        for rule, _, _ in RULES + [SHRUNK]:
            hb, ood, cos = [], [], []
            for s in seeds:
                e = (budget_point(s, k) if rule == "shrunk"
                     else s["budget"][str(k)].get(rule))
                if e is None:
                    continue
                hb.append(e["auc"])
                ood.append(np.mean([e["transfer"][f] for f in fams]))
                cos.append(e["cos_dense"])
            for m, v in (("hb", hb), ("ood", ood), ("cos", cos)):
                # a budget no seed's path lands on (k=1 is not always reachable)
                if not v:
                    per[rule][m].append((np.nan, np.nan, np.nan, 0))
                    continue
                sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
                per[rule][m].append((np.mean(v), sd / np.sqrt(len(v)), sd, len(v)))
    return {r: {m: np.array(v) for m, v in d.items()} for r, d in per.items()}


def dense_ref(rec):
    fams, seeds = rec["families"], rec["seeds"]
    hb = np.array([s["dense"]["auc"] for s in seeds])
    ood = np.array([np.mean([s["dense"]["transfer"][f] for f in fams]) for s in seeds])
    return hb, ood


def fig_curve(rec, cur, name="fig_lasso_curve"):
    ks = np.array(rec["kgrid"], float)
    hb_d, ood_d = dense_ref(rec)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=220)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.95, wspace=0.26)
    panels = [("hb", "Held-out AUC (HarmBench)", hb_d),
              ("ood", "Mean OOD transfer AUC", ood_d),
              ("cos", "|cos| with the dense direction", None)]
    for ax, (metric, ylab, ref) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(SPINE)
            ax.spines[s].set_linewidth(0.9)
        ax.grid(color=GRID, linewidth=0.7)
        ax.tick_params(colors=TEXT_M, labelsize=7.4, length=0)
        if ref is not None:
            ax.axhline(ref.mean(), color=C_DENSE, lw=1.0, ls=(0, (3.0, 2.0)), zorder=2)
            # at the left end the dashed line has the most room under it
            t = ax.text(ks[0], ref.mean(), "all 229", ha="left", va="bottom",
                        fontsize=6.3, color=C_DENSE)
            halo(t, 2.5)
        m = cur[SHRUNK[0]][metric][:, 0]
        ax.plot(ks, m, color=SHRUNK[2], lw=1.1, ls=(0, (1.6, 1.6)), zorder=3.5,
                alpha=0.8)
        for rule, lab, c in RULES:
            m, sem = cur[rule][metric][:, 0], cur[rule][metric][:, 1]
            ax.fill_between(ks, m - sem, m + sem, color=c, alpha=0.16, lw=0, zorder=3)
            ax.plot(ks, m, color=c, lw=1.6, zorder=4, solid_capstyle="round")
            ax.plot(ks, m, ls="", marker="o", ms=2.6, color=c, mec="white", mew=0.4,
                    zorder=5)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 50, 100, 229])
        ax.set_xticklabels(["1", "2", "5", "10", "25", "50", "100", "229"])
        ax.minorticks_off()
        ax.set_ylabel(ylab, fontsize=8.0, color=TEXT_M)
    fig.text(0.527, 0.035, "Traits retained ($k$)", ha="center", fontsize=8.8,
             color=TEXT_M)
    handles = [Line2D([0], [0], color=c, lw=1.8, label=lab) for _, lab, c in RULES]
    handles.append(Line2D([0], [0], color=SHRUNK[2], lw=1.1, ls=(0, (1.6, 1.6)),
                          label=SHRUNK[1]))
    handles.append(Line2D([0], [0], color=C_DENSE, lw=1.0, ls=(0, (3.0, 2.0)),
                          label="Dense fit, all 229 traits"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.2,
                     framealpha=0.95, edgecolor=SPINE, borderpad=0.35,
                     handletextpad=0.5, columnspacing=1.4, handlelength=1.8,
                     bbox_to_anchor=(0.527, -0.05))
    leg.get_frame().set_facecolor("white")
    save(fig, name)


def trait_stats(rec, k_sel=10):
    """Selection frequency at a budget of k_sel, coefficient sign, and the
    smallest budget at which each trait enters the path."""
    names, seeds = rec["names"], rec["seeds"]
    n = len(names)
    freq = np.zeros(n)
    coef_sum = np.zeros(n)
    entry = [[] for _ in range(n)]
    for s in seeds:
        e = budget_point(s, k_sel)
        if e:
            for j, c in zip(e["support"], e["coef"]):
                freq[j] += 1
                coef_sum[j] += np.sign(c)
        first = {}
        for pt in sorted(s["path"], key=lambda p: p["k"]):
            for j in pt["support"]:
                first.setdefault(j, pt["k"])
        for j in range(n):
            entry[j].append(first.get(j, np.nan))
    freq /= len(seeds)
    med = np.array([np.nanmedian(e) if np.any(~np.isnan(e)) else np.nan for e in entry])
    return freq, np.sign(coef_sum), med


def fig_traits(rec, k_sel=10, n_top=18, name="fig_lasso_traits"):
    names, n_seeds = rec["names"], len(rec["seeds"])
    freq, sign, med = trait_stats(rec, k_sel)
    idx = np.argsort(-freq)[:n_top][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), dpi=220,
                             gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.06},
                             sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.135, top=0.885)
    ypos = np.arange(len(idx))
    ax = axes[0]
    for ax_ in axes:
        ax_.set_facecolor("white")
        ax_.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax_.spines[s].set_visible(False)
        ax_.spines["bottom"].set_color(SPINE)
        ax_.spines["bottom"].set_linewidth(0.9)
        ax_.grid(axis="x", color=GRID, linewidth=0.7)
        ax_.tick_params(colors=TEXT_M, labelsize=7.4, length=0)
    for y, j in zip(ypos, idx):
        c = C_POS if sign[j] >= 0 else C_NEG
        ax.barh(y, freq[j] * 100, height=0.62, color=c, zorder=3)
        t = ax.text(freq[j] * 100 + 1.6, y, f"{freq[j] * 100:.0f}", va="center",
                    ha="left", fontsize=6.4, color=TEXT_M)
        halo(t, 2.0)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[j].replace("_", " ") for j in idx], fontsize=7.6)
    ax.set_ylim(-0.8, len(idx) - 0.2)
    ax.set_xlabel(f"Selected in % of the {n_seeds} splits (budget $k={k_sel}$)",
                  fontsize=8.2, color=TEXT_M)
    ax2 = axes[1]
    for y, j in zip(ypos, idx):
        if np.isnan(med[j]):
            continue
        c = C_POS if sign[j] >= 0 else C_NEG
        ax2.plot([1, med[j]], [y, y], color=c, lw=1.0, alpha=0.35, zorder=3)
        ax2.scatter([med[j]], [y], s=20, color=c, edgecolor="white", linewidth=0.5,
                    zorder=5)
    ax2.set_xscale("log")
    ax2.set_xlim(0.85, 260)
    ax2.set_xticks([1, 2, 5, 10, 25, 50, 100, 229])
    ax2.set_xticklabels(["1", "2", "5", "10", "25", "50", "100", "229"])
    ax2.minorticks_off()
    ax2.set_xlabel("Median budget at which the trait enters", fontsize=8.2,
                   color=TEXT_M)
    handles = [Line2D([0], [0], marker="s", ls="", color=C_POS, markersize=5,
                      label="pushes toward jailbroken"),
               Line2D([0], [0], marker="s", ls="", color=C_NEG, markersize=5,
                      label="pushes toward refused")]
    leg = fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=7.2,
                     framealpha=0.95, edgecolor=SPINE, borderpad=0.35,
                     handletextpad=0.4, columnspacing=1.4,
                     bbox_to_anchor=(0.61, 0.995))
    leg.get_frame().set_facecolor("white")
    save(fig, name)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  {name}.pdf/.png")


def summary(rec, cur):
    ks, fams = rec["kgrid"], rec["families"]
    hb_d, ood_d = dense_ref(rec)
    print(f"\n{len(rec['seeds'])} pool splits; test n="
          f"{rec['seeds'][0]['n_test']} ({rec['seeds'][0]['n_test_pos']} jb on seed 0); "
          f"transfer {', '.join(fams)}")
    print(f"dense 229-trait fit: HB {hb_d.mean():.3f} (sd {hb_d.std(ddof=1):.3f}), "
          f"OOD {ood_d.mean():.3f} (sd {ood_d.std(ddof=1):.3f})\n")
    cols = RULES + [SHRUNK]
    hdr = f"{'k':>4} |" + "".join(f"{r:>9s}" for r, _, _ in cols) + " |"
    hdr += "".join(f"{r + ' OOD':>12s}" for r, _, _ in cols) + " |  cos(dense)"
    print(hdr)
    for i, k in enumerate(ks):
        line = f"{k:>4} |"
        for rule, _, _ in cols:
            line += f"{cur[rule]['hb'][i, 0]:>9.3f}"
        line += " |"
        for rule, _, _ in cols:
            line += f"{cur[rule]['ood'][i, 0]:>12.3f}"
        line += f" |  {cur['lasso']['cos'][i, 0]:.3f}"
        print(line)
    # paired across splits: the same split scores both rules, so the difference
    # is far better determined than either curve
    print("\nlasso - univar, paired over splits (mean [95% CI of the mean]):")
    for k in (1, 3, 5, 10, 25, 50):
        if k not in ks:
            continue
        d = np.array([s_["budget"][str(k)]["lasso"]["auc"]
                      - s_["budget"][str(k)]["univar"]["auc"]
                      for s_ in rec["seeds"] if "lasso" in s_["budget"][str(k)]])
        sem = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  k={k:>3}  {d.mean():+.4f} [{d.mean() - 1.96 * sem:+.4f}, "
              f"{d.mean() + 1.96 * sem:+.4f}]  n={len(d)}")

    # per split, the smallest budget that gets within a fraction of that split's
    # own dense fit -- the across-seed mean can hide how variable the knee is
    for frac in (0.95, 0.99):
        need = []
        for s_, d in zip(rec["seeds"], hb_d):
            hit = [k for k in ks
                   if s_["budget"][str(k)].get("lasso", {"auc": 0.0})["auc"] >= frac * d]
            need.append(hit[0] if hit else np.nan)
        need = np.array(need, float)
        q = np.nanpercentile(need, [25, 50, 75])
        print(f"\nsmallest k reaching {frac:.0%} of its own split's dense AUC: "
              f"median {q[1]:.0f} (IQR {q[0]:.0f}-{q[2]:.0f}), "
              f"never on {int(np.isnan(need).sum())}/{len(need)} splits")

    # smallest budget whose mean reaches the dense fit
    for metric, ref in (("hb", hb_d.mean()), ("ood", ood_d.mean())):
        m = cur["lasso"][metric][:, 0]
        hit = [k for k, v in zip(ks, m) if v >= ref]
        print(f"\nlasso {metric}: reaches the dense fit ({ref:.3f}) at k="
              f"{hit[0] if hit else 'never'}; best {np.nanmax(m):.3f} "
              f"at k={ks[int(np.nanargmax(m))]}")


def print_selected(rec, k_sel, n_top=12):
    """The retained set, plus how interchangeable its members are: the trait
    vectors are far from orthogonal, so L1 picks one member of a correlated
    cluster and the choice moves between splits."""
    names = rec["names"]
    freq, sign, med = trait_stats(rec, k_sel)
    idx = np.argsort(-freq)[:n_top]
    print(f"\ntop {n_top} traits at budget k={k_sel} "
          f"(share of {len(rec['seeds'])} splits, sign, median entry budget):")
    for j in idx:
        side = "+jailbroken" if sign[j] >= 0 else "-refused   "
        enters = "never" if np.isnan(med[j]) else f"k={med[j]:.0f}"
        print(f"  {names[j]:24s} {freq[j] * 100:5.0f}%  {side}  enters at {enters}")
    vp = REPO / "full_trait_output/trait_matrix_layer16.npy"
    if vp.exists():
        V = np.load(vp).astype(np.float64)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        G = np.abs(V[idx] @ V[idx].T)
        np.fill_diagonal(G, 0.0)
        i, j2 = np.unravel_index(G.argmax(), G.shape)
        print(f"  most aligned pair among them: {names[idx[i]]} / {names[idx[j2]]} "
              f"|cos|={G.max():.2f}; mean |cos| {G[G > 0].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(REPO / "results/lasso_sweep.json"))
    ap.add_argument("--k_sel", type=int, default=10)
    args = ap.parse_args()
    rec = json.load(open(args.cells))
    cur = curves(rec)
    print("building:")
    print_selected(rec, args.k_sel)
    fig_curve(rec, cur)
    fig_traits(rec, args.k_sel)
    summary(rec, cur)


if __name__ == "__main__":
    main()
