#!/usr/bin/env python3
"""
Persona-removal steering on HarmBench, JBB and WildJB (official Llama snapshot, cumulative alpha 0.1-0.5).
  HarmBench  originally-refused pool n=300 seed 0. Alpaca arms from steer_v2_results/ (earlier runs), persona and
             random_full_s0 from steer_bench/harmbench_*.jsonl (run_steer_bench.sh hb).
  JBB        the 100 raw goals (run_steer_bench.sh jbb).
  WildJB     first 600 of the published 2000-behaviour sample, restricted to behaviours the model refused in the
             detection run (steer_bench/wjb_classified_responses.jsonl, jailbroken == False), as in the paper.
Dot = ASR, trace = 95% CI from a behaviour-clustered bootstrap (2000 reps); paired changes vs the original direction
use the same resampled behaviours for both arms. Two scorings: the harness's (word-5-gram degeneracy filter) and
corrected (also failing responses whose first 300 characters are repetitive junk, nopersona_steer_results.char_degenerate).
Writes figs/fig_steer_nopersona_bench{,_degen}.pdf/png, steer_bench/steer_nopersona_bench.json and
steer_bench/steer_nopersona_bench_table.tex (paired changes, both scorings).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent; OLD = HERE / "steer_v2_results"; NEW = HERE / "steer_bench"
sys.path.insert(0, str(HERE.parents[1] / "full_trait_tools"))
from plot_paper_result_figs import GRID, SPINE, TEXT_M, halo  # noqa: E402  (also sets the paper rcParams)
exec(compile(open(HERE / "nopersona_steer_results.py").read().split("def load")[0], str(HERE / "nopersona_steer_results.py"), "exec"))

C_REF, C_ORIG, C_PER, C_NR, C_RND = "#555555", "#1D5B8C", "#A6403A", "#9AA4AD", "#C9C9C9"
HB_OLD = {"alpacajb_d2": "alpaca1_alpacajb_d2", "alpacajb_d2_nopersona": "nopersona_det_alpacajb_d2_nopersona",
          "alpacajb_d2_nonr": "nopersona_det_alpacajb_d2_nonr", "sel_pca_1": "alpaca2_sel_pca_1",
          "sel_pca_1_nopersona": "nopersona_sel_sel_pca_1_nopersona", "sel_acts_1": "alpaca2_sel_acts_1",
          "sel_acts_1_nopersona": "nopersona_sel_sel_acts_1_nopersona"}
ROWS = [  # (group, label, arm, colour, original arm)
    (None, "PersonaJB direction", "persona", C_REF, None),
    ("Refusal direction", "original (paper run)", "paper:refusal", C_ORIG, None),
    ("Refusal direction", "persona removed", "refusal_nopersona", C_PER, "paper:refusal"),
    ("Jailbreak direction", "original (paper run)", "paper:dmu", C_ORIG, None),
    ("Jailbreak direction", "persona removed", "dmu_nopersona", C_PER, "paper:dmu"),
    ("Alpaca detector", "original", "alpacajb_d2", C_ORIG, None),
    ("Alpaca detector", "persona removed", "alpacajb_d2_nopersona", C_PER, "alpacajb_d2"),
    ("Top Alpaca PC", "original", "sel_pca_1", C_ORIG, None),
    ("Top Alpaca PC", "persona removed", "sel_pca_1_nopersona", C_PER, "sel_pca_1"),
    ("Top Alpaca prompt", "original", "sel_acts_1", C_ORIG, None),
    ("Top Alpaca prompt", "persona removed", "sel_acts_1_nopersona", C_PER, "sel_acts_1"),
    (None, "Random direction", "random_full_s0", C_RND, None)]
BENCH = [("harmbench", "HarmBench"), ("jbb", "JBB"), ("wildjb", "WildJB")]
wjb_refused = {json.loads(l)["behavior_id"] for l in open(NEW / "wjb_classified_responses.jsonl") if not json.loads(l)["jailbroken"]}

def path(bench, arm):
    if bench == "harmbench" and arm in HB_OLD: return OLD / f"{HB_OLD[arm]}.jsonl"
    return NEW / f"{bench}_{arm}.jsonl"

def load_arm(bench, arm):
    D = {}
    for l in open(path(bench, arm)):
        r = json.loads(l)
        if bench == "wildjb" and r["behavior_id"] not in wjb_refused: continue
        jb = bool(r["final_jailbroken"])
        D[r["pair_id"]] = (r["behavior_id"], jb, jb and not char_degenerate(r["final_response"] or ""))
    return D

data, boots, ids_of = {}, {}, {}
for b, _ in BENCH:
    data[b] = {r[2]: load_arm(b, r[2]) for r in ROWS if not r[2].startswith("paper:")}
    ids = sorted(data[b]["persona"]); assert all(sorted(a) == ids for a in data[b].values()), b
    ids_of[b] = ids
    beh = np.array([data[b]["persona"][i][0] for i in ids]); ub = np.unique(beh)
    rows_of = {u: np.where(beh == u)[0] for u in ub}; rng = np.random.default_rng(0)
    boots[b] = [np.concatenate([rows_of[u] for u in rng.choice(ub, len(ub))]) for _ in range(2000)]
vec = lambda b, arm, k: np.array([data[b][arm][i][k] for i in ids_of[b]], float)
ci = lambda b, x: np.percentile([x[s].mean() for s in boots[b]], [2.5, 97.5]) * 100

# intact refusal / jailbreak directions: the paper's runs (results/steering_cell_cis.json for HarmBench, all 1935
# originally refused pairs, and JBB; results/wjb_refused_cells.json for WildJB, the 975 refused of 2000 behaviours),
# on the model copy without the system-header chat template; no per-prompt pairing with the runs above
_C = json.load(open(HERE.parents[1] / "results/steering_cell_cis.json")); _W = json.load(open(HERE.parents[1] / "results/wjb_refused_cells.json"))
PAPER = {}
for _a in ("refusal", "dmu"):
    for _b in ("harmbench", "jbb"):
        _d = _C[f"{_a}|{_b}"]; PAPER[f"{_a}|{_b}"] = dict(n=_d["n_pairs"], asr=100 * _d["asr"], lo=100 * _d["lo"], hi=100 * _d["hi"], raw=100 * _d["asr"], paper=True)
    _d = _W[_a]; PAPER[f"{_a}|wildjb"] = dict(n=_d["n_refused"], asr=100 * _d["asr_refused"], lo=100 * _d["lo"], hi=100 * _d["hi"], raw=100 * _d["asr_refused"], paper=True)
stats = {}
for k, sc in ((1, "harness"), (2, "corrected")):
    for b, _ in BENCH:
        for g, lab, arm, c, o in ROWS:
            if arm.startswith("paper:"):
                stats[f"{sc}|{b}|{arm}"] = PAPER[f"{arm[6:]}|{b}"]; continue
            x = vec(b, arm, k); lo, hi = ci(b, x)
            st = dict(n=len(x), asr=100 * x.mean(), lo=lo, hi=hi, raw=100 * vec(b, arm, 1).mean())
            if o and not o.startswith("paper:"):
                d = x - vec(b, o, k); dlo, dhi = ci(b, d); st.update(diff=100 * d.mean(), diff_lo=dlo, diff_hi=dhi)
            stats[f"{sc}|{b}|{arm}"] = st

def figure(sc, out, degen):
    y, ys, gy = 0.0, [], {}
    for i, r in enumerate(ROWS):
        if i and r[0] != ROWS[i - 1][0]: y += 0.9 if r[0] else 0.5
        ys.append(-y); y += 1
        if r[0]: gy.setdefault(r[0], []).append(ys[-1])
    fig, axs = plt.subplots(1, 3, figsize=(397 / 72, 4.0), dpi=200, sharey=True)
    for ax, (b, title) in zip(axs, BENCH):
        for s in ax.spines.values(): s.set_color(SPINE); s.set_linewidth(0.9)
        ax.spines[["top", "right"]].set_visible(False)
        for yy, (g, lab, arm, c, o) in zip(ys, ROWS):
            st = stats[f"{sc}|{b}|{arm}"]
            if degen and st["raw"] > st["asr"] + 1e-9:
                ax.barh(yy, st["raw"] - st["asr"], left=st["asr"], height=0.42, color="white", edgecolor=c, hatch="////", lw=0.5, zorder=2)
            if o:
                ax.plot([stats[f"{sc}|{b}|{o}"]["asr"], st["asr"]], [yy, yy], color=c, lw=0.7, ls=(0, (2, 2)), zorder=1)
            ax.plot([st["lo"], st["hi"]], [yy, yy], color=c, lw=4, alpha=0.28, solid_capstyle="round", zorder=3)
            if st.get("paper"): ax.scatter([st["asr"]], [yy], s=20, facecolor="white", edgecolor=c, zorder=4, lw=1.1)
            else: ax.scatter([st["asr"]], [yy], s=22, color=c, zorder=4, edgecolor="white", lw=0.6)
            t = ax.text(st["asr"], yy + 0.28, f"{st['asr']:.0f}", ha="center", va="bottom", fontsize=5.6, color=TEXT_M, zorder=5); halo(t, 1.8)
        n = stats[f"{sc}|{b}|persona"]["n"]  # the runs on this page; paper rows use their own pools
        ax.set_title(f"{title} (n={n})", fontsize=7.8, color=TEXT_M, pad=4)
        ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75])
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=TEXT_M, labelsize=6.3, length=0); ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(ys); axs[0].set_yticklabels([r[1] for r in ROWS], fontsize=6.6, color=TEXT_M)
    for g, yl in gy.items():
        axs[0].text(-0.03, max(yl) + 0.72, g, transform=axs[0].get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.8, color="#333333", fontweight="bold")
    axs[0].set_ylim(ys[-1] - 0.7, ys[0] + 0.8)
    fig.supxlabel("attack success rate (%)", fontsize=7.2, color=TEXT_M, y=0.12 if degen else 0.11)
    h = [Line2D([], [], marker="o", ls="", color=C_ORIG, label="original direction"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=C_ORIG, label="original, paper run"),
         Line2D([], [], marker="o", ls="", color=C_PER, label="persona component removed")]
    if degen: h.append(Patch(facecolor="white", edgecolor=TEXT_M, hatch="////", lw=0.5, label="judged jailbroken, but repetitive junk"))
    fig.legend(handles=h, loc="lower center", ncol=3 if degen else 2, fontsize=6.3, frameon=False, handletextpad=0.3, columnspacing=1.0)  # noqa
    fig.subplots_adjust(left=0.265, right=0.985, top=0.93, bottom=0.2 if degen else 0.19, wspace=0.13)
    for e in ("pdf", "png"): fig.savefig(HERE / f"figs/{out}.{e}", dpi=200)

figure("harness", "fig_steer_nopersona_bench", False)
figure("corrected", "fig_steer_nopersona_bench_degen", True)
json.dump(stats, open(NEW / "steer_nopersona_bench.json", "w"), indent=1)

# paired-change table, both scorings
PAIRS = [("Alpaca detector, persona removed", "alpacajb_d2_nopersona"),
         ("Top Alpaca PC, persona removed", "sel_pca_1_nopersona"), ("Top Alpaca prompt, persona removed", "sel_acts_1_nopersona")]
cell = lambda st: f"${st['diff']:+.1f}$ {{\\scriptsize $[{st['diff_lo']:+.1f}, {st['diff_hi']:+.1f}]$}}".replace("+-", "-")
L = []
for sc, head in (("harness", "Harness scoring"), ("corrected", "Repetitive junk counted as failure")):
    L.append(f"    \\multicolumn{{4}}{{l}}{{\\emph{{{head}}}}} \\\\")
    for lab, arm in PAIRS:
        L.append(f"    {lab} & " + " & ".join(cell(stats[f"{sc}|{b}|{arm}"]) for b, _ in BENCH) + " \\\\")
    if sc == "harness": L.append("    \\midrule")
(NEW / "steer_nopersona_bench_table.tex").write_text("\n".join(L) + "\n")
for k, v in stats.items():
    if "diff" in v: print(f"{k:45s} {v['asr']:5.1f}  diff {v['diff']:+5.1f} [{v['diff_lo']:+.1f}, {v['diff_hi']:+.1f}]")
    else: print(f"{k:45s} {v['asr']:5.1f}  n={v['n']}")
