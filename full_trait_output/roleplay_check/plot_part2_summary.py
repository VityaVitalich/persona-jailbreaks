#!/usr/bin/env python3
"""Summary figure for the synthetic role-play check (Part 2).
(a) PersonaJB score per prompt group: harmful vs benign request x framing (plain + 3 role-play
    types). Dot = mean over prompts of the seed-averaged P(jb); bar = interquartile range.
(b) Aggregate separation (folded AUC, 95% behaviour-bootstrap CI) along the two axes, on the
    same 4,600 prompts: harmful vs benign, and role-play vs plain.
Reads part2_scores.npz + part2_results.json; writes part2_summary.pdf (+ .png preview)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
HARM, BEN = "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

d = np.load(HERE / "part2_scores.npz", allow_pickle=True)
rows = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "synthetic_prompts.jsonl")}
meta = [rows[i] for i in d["ids"]]; s = d["scores"].mean(0)
res = json.load(open(HERE / "part2_results.json"))["all"]
FT = [("plain", "Plain\nrequest"), ("neutral_persona", "Persona"), ("neutral_fiction", "Fiction"), ("jb_persona", "Jailbreak-\nstyle persona")]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.35, 1]})

# (a) per-group score
x = np.arange(len(FT))
for harm, col, mk, dx in [("harmful", HARM, "o", -0.09), ("benign", BEN, "s", 0.09)]:
    mean, lo, hi = [], [], []
    for ft, _ in FT:
        v = s[[i for i, r in enumerate(meta) if r["harm"] == harm and r["frame_type"] == ft]]
        mean.append(v.mean()); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    a.vlines(x + dx, lo, hi, color=col, lw=2, alpha=.45)
    a.plot(x + dx, mean, color=col, lw=2, marker=mk, ms=7, mec="white", mew=1.5, zorder=3)
    a.text(x[-1] + dx + 0.18, mean[-1], harm.capitalize(), color=INK, va="center", fontsize=9)
a.axvspan(0.5, 3.5, color=GRID, alpha=.35, lw=0, zorder=0)
a.text(2, 0.965, "role-play framings", ha="center", va="top", color=MUTED, fontsize=8)
a.set_xticks(x, [l for _, l in FT]); a.set_xlim(-0.4, 3.85); a.set_ylim(0, 1.0); a.set_yticks([0, .25, .5, .75, 1])
a.set_ylabel("Detector score P(jb)"); a.yaxis.grid(True, color=GRID, lw=.8); a.set_axisbelow(True)
a.set_title("(a) Score by prompt group", loc="left", fontsize=9.5, color=INK)

# (b) aggregate separation
def fold(o):
    v, (l, h) = o["auc"], o["ci"]
    return (v, l, h) if v >= .5 else (1 - v, 1 - h, 1 - l)
bars = [("Harmful vs benign", fold(res["axis_harm_vs_benign"]), INK),
        ("Harmful vs benign,\nrole-play prompts only", fold(res["harm_vs_benign"]["all_roleplay"]), INK),
        ("Role-play vs plain", fold(res["axis_roleplay_vs_plain"]), MUTED)]
y = np.arange(len(bars))[::-1]
for yi, (lab, (v, l, h), col) in zip(y, bars):
    b.plot([l, h], [yi, yi], color=col, lw=2.5, solid_capstyle="round")
    b.plot(v, yi, "o", color=col, ms=8, mec="white", mew=1.5, zorder=3)
    b.text(h + 0.015, yi, f"{v:.2f}", va="center", color=INK, fontsize=9)
b.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
b.text(0.505, y[0] + 0.55, "chance", color=MUTED, fontsize=8, va="bottom")
b.set_yticks(y, [l for l, _, _ in bars]); b.set_ylim(-0.6, len(bars) - 0.2); b.set_xlim(0.45, 1.0)
b.set_xlabel("Separation (AUC, direction-free)"); b.xaxis.grid(True, color=GRID, lw=.8); b.set_axisbelow(True)
b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
b.set_title("(b) What the score separates", loc="left", fontsize=9.5, color=INK)

fig.tight_layout(w_pad=2)
fig.savefig(HERE / "part2_summary.pdf", bbox_inches="tight")
fig.savefig(HERE / "part2_summary.png", dpi=200, bbox_inches="tight")
print("wrote part2_summary.pdf/.png")
