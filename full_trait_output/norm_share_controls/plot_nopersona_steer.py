#!/usr/bin/env python3
"""
Persona-removed steering figures (official Llama snapshot, HarmBench originally-refused n=300 seed 0, cumulative
alpha 0.1-0.5). Rows: each steering direction as run, then with its persona component removed (and, for the
detector, with a No Robots activation span removed instead). Dot = ASR, trace = 95% CI from a behaviour-clustered
bootstrap (2000 reps); right column = paired change vs the original direction, same bootstrap (the same
behaviours resampled for both arms).
  fig_steer_nopersona        ASR as the harness scores it (its word-5-gram degeneracy filter)
  fig_steer_nopersona_degen  ASR after also failing responses whose first 300 characters are repetitive junk
                             (nopersona_steer_results.char_degenerate); the hatched segment is what that removes
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
HERE = Path(__file__).resolve().parent; R = HERE / "steer_v2_results"
sys.path.insert(0, str(HERE.parents[1] / "full_trait_tools"))
from plot_paper_result_figs import GRID, SPINE, TEXT_M, halo  # noqa: E402  (also sets the paper rcParams)
exec(compile(open(HERE / "nopersona_steer_results.py").read().split("def load")[0], str(HERE / "nopersona_steer_results.py"), "exec"))

C_ORIG, C_PER, C_NR = "#1D5B8C", "#A6403A", "#9AA4AD"
GROUPS = [("Alpaca detector (draw 2)", "alpaca1_alpacajb_d2",
           [("persona removed", "nopersona_det_alpacajb_d2_nopersona", C_PER),
            ("No Robots span removed", "nopersona_det_alpacajb_d2_nonr", C_NR)]),
          ("Top Alpaca PC", "alpaca2_sel_pca_1", [("persona removed", "nopersona_sel_sel_pca_1_nopersona", C_PER)]),
          ("Top Alpaca prompt", "alpaca2_sel_acts_1", [("persona removed", "nopersona_sel_sel_acts_1_nopersona", C_PER)])]

def load_arm(name):
    D = {}
    for l in open(R / f"{name}.jsonl"):
        r = json.loads(l); jb = bool(r["final_jailbroken"])
        D[r["pair_id"]] = (r["behavior_id"], jb, jb and not char_degenerate(r["final_response"] or ""))
    return D
arms = {n: load_arm(n) for _, o, sub in GROUPS for n in [o] + [s[1] for s in sub]}
ids = sorted(arms["alpaca1_alpacajb_d2"]); assert all(sorted(a) == ids for a in arms.values()) and len(ids) == 300
beh = np.array([arms["alpaca1_alpacajb_d2"][i][0] for i in ids]); ub = np.unique(beh)
rows_of = {b: np.where(beh == b)[0] for b in ub}
rng = np.random.default_rng(0)
BOOT = [np.concatenate([rows_of[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
vec = lambda n, k: np.array([arms[n][i][k] for i in ids], float)  # k=1 harness, k=2 corrected
ci = lambda x: np.percentile([x[b].mean() for b in BOOT], [2.5, 97.5]) * 100

def figure(k, out, degen):
    rows = []  # (label, group, arm, colour, original arm or None)
    for g, o, sub in GROUPS:
        rows.append(("original direction", g, o, C_ORIG, None))
        rows += [(lab, g, n, c, o) for lab, n, c in sub]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for s in ax.spines.values(): s.set_color(SPINE); s.set_linewidth(0.9)
    ax.spines[["top", "right"]].set_visible(False)
    y, ys, gy = 0.0, [], {}
    for i, r in enumerate(rows):
        if i and r[1] != rows[i - 1][1]: y += 1.0
        ys.append(y); gy.setdefault(r[1], []).append(y); y += 1
    ys = -np.array(ys)
    stats = {}
    for yy, (lab, g, n, c, o) in zip(ys, rows):
        x = vec(n, k); m = 100 * x.mean(); lo, hi = ci(x)
        if degen:
            raw = 100 * vec(n, 1).mean()
            ax.barh(yy, raw - m, left=m, height=0.36, color="white", edgecolor=c, hatch="////", lw=0.6, zorder=2, alpha=0.9)
        ax.plot([lo, hi], [yy, yy], color=c, lw=4.5, alpha=0.25, solid_capstyle="round", zorder=3)
        ax.scatter([m], [yy], s=34, color=c, zorder=4, edgecolor="white", lw=0.7)
        t = ax.text(m, yy + 0.30, f"{m:.1f}", ha="center", va="bottom", fontsize=6.6, color=TEXT_M, zorder=5); halo(t)
        stats[n] = dict(asr=m, lo=lo, hi=hi)
        if o:
            d = (x - vec(o, k)); dm = 100 * d.mean(); dlo, dhi = ci(d)
            stats[n].update(diff=dm, diff_lo=dlo, diff_hi=dhi)
            ax.annotate("", xy=(m, yy), xytext=(100 * vec(o, k).mean(), yy), zorder=1,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8, ls=(0, (2, 2))))
            ax.text(1.015, yy, f"{dm:+.1f} pp  [{dlo:+.1f}, {dhi:+.1f}]", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=6.8, color=TEXT_M)
    ax.text(1.015, ys[0] + 0.75, "change vs original [95% CI]", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6.8, color=TEXT_M, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7.4, color=TEXT_M)
    for g, yl in gy.items():
        ax.text(-0.02, -yl[0] + 0.75, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.6, color="#333333", fontweight="bold")
    ax.set_xlim(0, 75); ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.2)
    ax.set_xlabel("attack success rate on originally refused HarmBench prompts (%)", fontsize=7.8, color=TEXT_M)
    ax.tick_params(axis="x", colors=TEXT_M, labelsize=7, length=0); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    h = [Line2D([], [], marker="o", ls="", color=C_ORIG, label="original direction"),
         Line2D([], [], marker="o", ls="", color=C_PER, label="persona component removed"),
         Line2D([], [], marker="o", ls="", color=C_NR, label="No Robots span removed (control)")]
    if degen: h.append(Patch(facecolor="white", edgecolor=TEXT_M, hatch="////", lw=0.6, label="judged jailbroken, but repetitive junk"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.35, -0.17), ncol=2, fontsize=6.6, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.27, right=0.70, top=0.97, bottom=0.27)
    for e in ("pdf", "png"): fig.savefig(HERE / f"figs/{out}.{e}", dpi=200)
    return stats

res = {"harness": figure(1, "fig_steer_nopersona", False), "corrected": figure(2, "fig_steer_nopersona_degen", True)}
json.dump(res, open(HERE / "figs/steer_nopersona_plotdata.json", "w"), indent=1)
for k, s in res.items():
    print(k); [print(f"  {n:40s} {v['asr']:5.1f} [{v['lo']:.1f}, {v['hi']:.1f}]" + (f"  diff {v['diff']:+.1f} [{v['diff_lo']:+.1f}, {v['diff_hi']:+.1f}]" if 'diff' in v else "")) for n, v in s.items()]
