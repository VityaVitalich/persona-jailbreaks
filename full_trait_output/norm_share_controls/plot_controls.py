#!/usr/bin/env python3
"""
Local figures for the subspace controls (Llama-3.1-8B, L16). Not the paper figure; fig_direction_story
in the paper is untouched.

  fig_direction_story_controls  left: share of squared norm of the refusal direction and the jailbreak
      direction (dmu, all data) in the persona span vs matched 229-dim subspaces; right: HarmBench test
      AUC of the projection onto each subspace (50 pool splits, 95% CI of the mean), full direction as a
      dashed line. Recomputed here from saved activations (fast; no LDA).
  fig_detector_controls  the PersonaJB detector (StandardScaler + LR C=10) on each 229-direction basis,
      from trait_pc_sweep/summary_llama_subspace_controls.json: HarmBench test, avg transfer folded
      (paper metric) and avg transfer signed.
Control families with 5 draws: dot = mean over draws, whisker = min-max over draws.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
from llama_50seed_span import load_direction  # noqa: E402

C_REF, C_DMU, C_CTRL, C_RAND = "#1D5B8C", "#C88A2D", "#6B6B6B", "#B5B5B5"
GRID, SPINE, TXT = "#E7E7E7", "#C9C9C9", "#555555"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
GROUPS = [("Persona span", ["persona"]), ("Alpaca PCA-229", ["alpaca_pca"]),
          ("229 Alpaca activations", [f"acts_d{i}" for i in range(5)]),
          ("229 shuffled-label contrasts", [f"noise_d{i}" for i in range(5)]),
          ("229 No Robots prompts", [f"nr_d{i}" for i in range(5)]),
          ("229 Wikipedia passages", [f"wiki_d{i}" for i in range(5)]),
          ("Random 229", [f"gauss_d{i}" for i in range(5)])]


def style(ax):
    ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.8)
    for s in ax.spines.values(): s.set_color(SPINE); s.set_linewidth(0.9)
    ax.tick_params(colors=TXT, length=0, pad=2)


def direction_data():
    cfg = S.CFG["llama"]
    T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = {"persona": T, **{k: v.astype(np.float64) for k, v in np.load(HERE / "control_bases_llama.npz").items()},
         **{k: v.astype(np.float64) for k, v in np.load(HERE / "extra_bases_llama.npz").items()}}
    hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    X, y, hv = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
    X = X.astype(np.float64)
    unit = lambda v: v / np.linalg.norm(v)
    ref = load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt")
    dmu_all = unit(X[y == 1].mean(0) - X[y == 0].mean(0))
    sp = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]
    dtr = [unit(X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)) for tr, _ in sp]
    D = {"full": {"ref": [roc_auc_score(y[te], -X[te] @ ref) for _, te in sp],
                  "dmu": [roc_auc_score(y[te], X[te] @ d) for (_, te), d in zip(sp, dtr)]}}
    for n, V in B.items():
        Q = np.linalg.qr(V.T)[0]; P = lambda d: Q @ (Q.T @ d)
        D[n] = {"share_ref": float(np.sum((Q.T @ ref) ** 2)), "share_dmu": float(np.sum((Q.T @ dmu_all) ** 2)),
                "ref": [roc_auc_score(y[te], -X[te] @ P(ref)) for _, te in sp],
                "dmu": [roc_auc_score(y[te], X[te] @ P(d)) for (_, te), d in zip(sp, dtr)]}
    json.dump(D, open(HERE / "direction_controls_data.json", "w"))
    return D


def agg(vals):
    v = np.array(vals); return v.mean(), v.min(), v.max()


def fig_direction(D):
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(6.6, 3.9), sharey=True, gridspec_kw={"width_ratios": [1, 0.8], "wspace": 0.08})
    pitch, gap = 0.62, 1.1
    ys, yt, ylab = {}, [], []
    y0 = 0.0
    for di, (dk, dname, col) in enumerate([("ref", "Refusal Direction", C_REF), ("dmu", "Jailbreak Direction", C_DMU)]):
        for gi, (gname, _) in enumerate(GROUPS):
            ys[(dk, gname)] = y0 - gi * pitch; yt.append(y0 - gi * pitch); ylab.append(gname)
        ys[(dk, "head")] = y0 + 0.62
        y0 -= len(GROUPS) * pitch + gap
    for ax in (aL, aR): style(ax)
    for dk, dname, col in [("ref", "Refusal Direction", C_REF), ("dmu", "Jailbreak Direction", C_DMU)]:
        aL.text(0.5, ys[(dk, "head")], dname, color=col, fontweight="bold", fontsize=9, va="center")
        full_m = np.mean(D["full"][dk]); yy = [ys[(dk, g)] for g, _ in GROUPS]
        aR.plot([full_m, full_m], [min(yy) - 0.3, max(yy) + 0.3], ls=(0, (3, 2)), color=col, lw=1.0, alpha=0.8)
        aR.text(full_m, max(yy) + 0.42, f"full {full_m:.3f}", color=col, fontsize=7.5, ha="center", va="center")
        for gname, ns in GROUPS:
            y = ys[(dk, gname)]; c = col if gname == "Persona span" else (C_RAND if gname == "Random 229" else C_CTRL)
            m, lo, hi = agg([100 * D[n][f"share_{dk}"] for n in ns])
            aL.barh(y, m, height=0.42, color=c, alpha=0.9 if gname == "Persona span" else 0.75, zorder=3)
            if len(ns) > 1: aL.plot([lo, hi], [y, y], color="#333", lw=1.0, zorder=4)
            aL.text(max(m, hi) + 1.0, y, f"{m:.1f}%", va="center", fontsize=7.5, color=TXT)
            per_basis = np.array([D[n][dk] for n in ns])          # [draws, splits]
            mm = per_basis.mean(1); ci = 1.96 * per_basis.std(1).mean() / np.sqrt(per_basis.shape[1])
            aR.plot([mm.mean() - ci, mm.mean() + ci], [y, y], color=c, lw=5, alpha=0.25, solid_capstyle="round")
            if len(ns) > 1: aR.plot([mm.min(), mm.max()], [y, y], color="#333", lw=1.0, zorder=4)
            aR.scatter([mm.mean()], [y], s=26 if gname != "Persona span" else 70, marker="*" if gname == "Persona span" else "o",
                       color=c, edgecolor="white", linewidth=0.5, zorder=5)
    aL.set_yticks(yt); aL.set_yticklabels(ylab, fontsize=8)
    aL.set_xlim(0, 62); aL.set_xlabel("share of squared norm in subspace (%)", color=TXT)
    aR.set_xlim(0.74, 0.90); aR.set_xticks([0.75, 0.80, 0.85, 0.90])
    aR.set_xlabel("HarmBench AUC of the projected direction", color=TXT)
    top = max(v for k, v in ys.items() if k[1] == "head") + 0.35
    aL.set_ylim(min(yt) - 0.45, top)
    fig.text(0.01, -0.03, "Llama-3.1-8B, layer 16. Every subspace has 229 dimensions. Control families: 5 draws, "
             "bar/dot = mean, whisker = min-max over draws. AUC band: 95% CI over 50 splits.", fontsize=6.5, color=TXT)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_direction_story_controls.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_detector():
    C = {**json.load(open(OUT / "trait_pc_sweep/summary_llama_subspace_controls.json"))["configs"],
         **json.load(open(OUT / "trait_pc_sweep/summary_llama_extra_controls.json"))["configs"]}
    fams = S.FAMS
    rows = [("PersonaJB (229 traits)", ["traits_k229"], C_REF), ("Persona span, orthonormal", ["svd_k229"], "#7FA6C6"),
            ("229 Alpaca activations", [f"acts_d{i}_k229" for i in range(5)], C_CTRL),
            ("229 shuffled-label contrasts", [f"noise_d{i}_k229" for i in range(5)], C_CTRL),
            ("229 No Robots prompts", [f"nr_d{i}_k229" for i in range(5)], C_CTRL),
            ("229 Wikipedia passages", [f"wiki_d{i}_k229" for i in range(5)], C_CTRL),
            ("Alpaca PCA-229", ["alpaca_pca_k229"], C_CTRL), ("HarmBench PCA-229 (no labels)", ["hb_pca_k229"], C_CTRL),
            ("Random 229", [f"gauss_d{i}_k229" for i in range(5)], C_RAND)]
    def per_seed(c, metric):
        ps = c["per_seed"]
        if metric == "hb": return np.array(ps["human_test"]["auc"])
        key = "auc" if metric == "xf" else "auc_signed"
        return np.nanmean(np.array([ps[f][key] for f in fams]), 0)
    panels = [("hb", "HarmBench test AUC", (0.55, 0.96)), ("xf", "Avg. transfer AUC, folded\n(paper metric)", (0.55, 0.83)),
              ("xs", "Avg. transfer AUC, signed", (0.55, 0.76))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    ys = -np.arange(len(rows), dtype=float)
    for ax, (mk, title, xl) in zip(axes, panels):
        style(ax)
        for y, (name, keys, col) in zip(ys, rows):
            vals = np.array([per_seed(C[k], mk) for k in keys])     # [draws, seeds]
            mm = vals.mean(1); ci = 1.96 * vals.std(1, ddof=1).mean() / np.sqrt(vals.shape[1])
            ax.plot([mm.mean() - ci, mm.mean() + ci], [y, y], color=col, lw=5, alpha=0.25, solid_capstyle="round")
            if len(keys) > 1: ax.plot([mm.min(), mm.max()], [y, y], color="#333", lw=1.0, zorder=4)
            ax.scatter([mm.mean()], [y], s=70 if name.startswith("PersonaJB") else 26, marker="*" if name.startswith("PersonaJB") else "o",
                       color=col, edgecolor="white", linewidth=0.5, zorder=5)
            ax.text(xl[1] - 0.004, y, f"{mm.mean():.3f}", fontsize=7, color=TXT, ha="right", va="center")
        ref = np.array([per_seed(C["traits_k229"], mk)]).mean()
        ax.axvline(ref, color=C_REF, lw=0.8, ls=(0, (3, 2)), alpha=0.7)
        ax.set_xlim(*xl); ax.set_xticks([t for t in (0.6, 0.7, 0.8, 0.9) if t < xl[1] - 0.04]); ax.set_title(title, fontsize=8.5, color=TXT)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([r[0] for r in rows], fontsize=8); axes[0].set_ylim(ys[-1] - 0.6, 0.6)
    fig.text(0.01, -0.02, "Same detector as PersonaJB (StandardScaler + logistic regression, C=10), 50 pool splits; only the 229 "
             "projection directions change. Dot = mean (over draws where 5), whisker = min-max over draws, band = 95% CI over splits.",
             fontsize=6.3, color=TXT)
    for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_detector_controls.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    fig_direction(direction_data()); fig_detector(); print("saved", HERE / "fig_direction_story_controls.pdf", HERE / "fig_detector_controls.pdf")
