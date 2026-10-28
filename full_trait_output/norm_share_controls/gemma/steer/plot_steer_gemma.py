#!/usr/bin/env python3
"""Gemma-4-31B steering figure in the style of fig_steer_nopersona_bench_degen (Llama): HarmBench, 300 refused
prompts (pool seed 0), cumulative alpha 0.1-0.5. Rows: PersonaJB direction (detector normal) and its activation
pattern in persona space; refusal and jailbreak directions, full / persona component only / persona component
removed; random direction. PersonaJB and random come from the earlier full-pool runs restricted to the same prompts
(open circles). Dot = ASR counting only coherent text (Gemma gibberish guard looks_like_text + word-level degeneracy);
hatched = judged jailbroken but gibberish; band = 95% CI, bootstrap over behaviours (2000). The persona-pattern arm
uses its alpha <= 0.5 results (results_upto0.5.jsonl). Writes ../figs/fig_steer_split_gemma.pdf/png and
steer_split_gemma.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent; O = HERE.parents[3]
sys.path.insert(0, str(O / "full_trait_tools"))
from plot_paper_result_figs import GRID, SPINE, TEXT_M, halo  # noqa: E402
from extract_gemma_steering_cells import looks_like_text  # noqa: E402
NEW = HERE / "outputs_gemma_split/harmbench"; OLD = O / "full_trait_output/outputs_gemma_steer/harmbench"
C_REF, C_ORIG, C_IN, C_PER, C_RND = "#555555", "#1D5B8C", "#5E9C76", "#A6403A", "#C9C9C9"
ROWS = [  # (group, label, file, colour, original, earlier run)
    ("PersonaJB direction", "detector normal", OLD / "w_persona__full/results.jsonl", C_REF, None, True),
    ("PersonaJB direction", "activation pattern, persona space", NEW / "persona_pattern/results_upto0.5.jsonl", C_IN, "PersonaJB direction|detector normal", False),
    ("Refusal direction", "original", NEW / "refusal/results.jsonl", C_ORIG, None, False),
    ("Refusal direction", "persona component only", NEW / "refusal_vin/results.jsonl", C_IN, "Refusal direction|original", False),
    ("Refusal direction", "persona component removed", NEW / "refusal_vout/results.jsonl", C_PER, "Refusal direction|original", False),
    ("Jailbreak direction", "original", NEW / "dmu/results.jsonl", C_ORIG, None, False),
    ("Jailbreak direction", "persona component only", NEW / "dmu_vin/results.jsonl", C_IN, "Jailbreak direction|original", False),
    ("Jailbreak direction", "persona component removed", NEW / "dmu_vout/results.jsonl", C_PER, "Jailbreak direction|original", False),
    (None, "Random direction", OLD / "random_full_s0__full/results.jsonl", C_RND, None, True)]
tb = lambda x: x is True or str(x) == "True"
ids = sorted(str(json.loads(l)["pair_id"]) for l in open(NEW / "refusal/results.jsonl"))
def load(p):
    D = {str(r["pair_id"]): r for r in map(json.loads, open(p))}
    rr = [D[i] for i in ids]
    raw = np.array([tb(r["final_jailbroken"]) for r in rr], float)
    ok = np.array([tb(r["final_jailbroken"]) and not tb(r.get("final_degenerate")) and looks_like_text(r.get("final_response", "")) for r in rr], float)
    return rr, raw, ok
data = {f"{g}|{l}" if g else l: load(p) for g, l, p, *_ in ROWS}
beh = np.array([r["behavior_id"] for r in data["Refusal direction|original"][0]]); ub = np.unique(beh)
rows_of = {u: np.where(beh == u)[0] for u in ub}; rng = np.random.default_rng(0)
boots = [np.concatenate([rows_of[u] for u in rng.choice(ub, len(ub))]) for _ in range(2000)]
ci = lambda x: np.percentile([x[s].mean() for s in boots], [2.5, 97.5]) * 100
stats = {}
for g, l, p, c, o, old in ROWS:
    k = f"{g}|{l}" if g else l; _, raw, ok = data[k]; lo, hi = ci(ok)
    st = dict(n=len(ok), asr=100 * ok.mean(), lo=lo, hi=hi, raw=100 * raw.mean(), earlier_run=old)
    if o:
        d = ok - data[o][2]; dlo, dhi = ci(d); st.update(vs=o, diff=100 * d.mean(), diff_lo=dlo, diff_hi=dhi)
    stats[k] = st
y, ys, gy = 0.0, [], {}
for i, r in enumerate(ROWS):
    if i and r[0] != ROWS[i - 1][0]: y += 0.9 if r[0] else 0.5
    ys.append(-y); y += 1
    if r[0]: gy.setdefault(r[0], []).append(ys[-1])
fig, ax = plt.subplots(figsize=(397 / 72 * 0.72, 3.05), dpi=200)
for s in ax.spines.values(): s.set_color(SPINE); s.set_linewidth(0.9)
ax.spines[["top", "right"]].set_visible(False)
for yy, (g, l, p, c, o, old) in zip(ys, ROWS):
    st = stats[f"{g}|{l}" if g else l]
    if st["raw"] > st["asr"] + 1e-9:
        ax.barh(yy, st["raw"] - st["asr"], left=st["asr"], height=0.42, color="white", edgecolor=c if c != C_RND else "#999999", hatch="////", lw=0.5, zorder=2)
    if o: ax.plot([stats[o]["asr"], st["asr"]], [yy, yy], color=c, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.plot([st["lo"], st["hi"]], [yy, yy], color=c, lw=4, alpha=0.28, solid_capstyle="round", zorder=3)
    if old: ax.scatter([st["asr"]], [yy], s=20, facecolor="white", edgecolor=c if c != C_RND else "#999999", zorder=4, lw=1.1)
    else: ax.scatter([st["asr"]], [yy], s=22, color=c, zorder=4, edgecolor="white", lw=0.6)
    t = ax.text(st["asr"], yy + 0.28, f"{st['asr']:.0f}", ha="center", va="bottom", fontsize=5.6, color=TEXT_M, zorder=5); halo(t, 1.8)
ax.set_title(f"HarmBench (n={len(ids)})", fontsize=7.8, color=TEXT_M, pad=4)
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.tick_params(axis="x", colors=TEXT_M, labelsize=6.3, length=0); ax.tick_params(axis="y", length=0)
ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in ROWS], fontsize=6.6, color=TEXT_M)
for g, yl in gy.items():
    ax.text(-0.03, max(yl) + 0.72, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.8, color="#333333", fontweight="bold")
ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.8); ax.set_xlabel("attack success rate (%)", fontsize=7.2, color=TEXT_M, labelpad=2)
h = [Line2D([], [], marker="o", ls="", markersize=4.5, color=C_ORIG, label="original direction"),
     Line2D([], [], marker="o", ls="", markersize=4.5, color=C_IN, label="persona component only"),
     Line2D([], [], marker="o", ls="", markersize=4.5, color=C_PER, label="persona component removed"),
     Line2D([], [], marker="o", ls="", markersize=4.5, markerfacecolor="white", markeredgecolor=C_REF, label="earlier run, same prompts"),
     Patch(facecolor="white", edgecolor=TEXT_M, hatch="////", lw=0.5, label="judged jailbroken, but gibberish")]
fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6.0, frameon=False, handletextpad=0.3, columnspacing=1.0)
fig.subplots_adjust(left=0.43, right=0.97, top=0.93, bottom=0.27)
for e in ("pdf", "png"): fig.savefig(HERE.parent / f"figs/fig_steer_split_gemma.{e}", dpi=200)
json.dump(stats, open(HERE / "steer_split_gemma.json", "w"), indent=1)
for k, v in stats.items(): print(f"{k:55s} {v['asr']:5.1f} raw {v['raw']:5.1f}" + (f"  diff {v['diff']:+5.1f} [{v['diff_lo']:+.1f}, {v['diff_hi']:+.1f}]" if "diff" in v else ""))
