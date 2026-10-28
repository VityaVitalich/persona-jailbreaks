#!/usr/bin/env python3
"""Paper figure (appendix, role-play check): what the PersonaJB score separates on the synthetic
JBB set. Folded AUC max(a,1-a), mean over 50 seeds, 95% behaviour-bootstrap CI. Reads
part2_results.json; writes fig_roleplay_separation.pdf (+ .png preview)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
res = json.load(open(HERE / "part2_results.json"))["all"]

def fold(o):
    v, (l, h) = o["auc"], o["ci"]
    return (v, l, h) if v >= .5 else (1 - v, 1 - h, 1 - l)

hb = res["harm_vs_benign"]
bars = [("Harmful vs. benign, all", fold(res["axis_harm_vs_benign"]), INK),
        ("    plain prompts", fold(hb["plain"]), INK),
        ("    role-play prompts", fold(hb["all_roleplay"]), INK),
        ("        persona", fold(hb["neutral_persona"]), MUTED),
        ("        fiction", fold(hb["neutral_fiction"]), MUTED),
        ("        jailbreak-style persona", fold(hb["jb_persona"]), MUTED),
        ("Role-play vs. plain", fold(res["axis_roleplay_vs_plain"]), INK)]
fig, b = plt.subplots(figsize=(4.4, 2.7))
y = np.array([7.6, 6.6, 5.6, 4.6, 3.6, 2.6, 1.0])  # gap before the role-play axis
for yi, (lab, (v, l, h), col) in zip(y, bars):
    b.plot([l, h], [yi, yi], color=col, lw=2.5, solid_capstyle="round")
    b.plot(v, yi, "o", color=col, ms=8, mec="white", mew=1.5, zorder=3)
    b.text(h + 0.015, yi, f"{v:.2f}", va="center", color=INK, fontsize=9)
b.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
b.text(0.507, y[0] + 0.45, "chance", color=MUTED, fontsize=8, va="bottom")
b.set_yticks(y, [l for l, _, _ in bars], ha="left"); b.tick_params(axis="y", pad=122); b.set_ylim(0.4, 8.5); b.set_xlim(0.45, 1.0); b.set_xticks([.5, .6, .7, .8, .9, 1.0])
b.set_xlabel("Separation (AUC)"); b.xaxis.grid(True, color=GRID, lw=.8); b.set_axisbelow(True)
b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "fig_roleplay_separation.pdf", bbox_inches="tight")
fig.savefig(HERE / "fig_roleplay_separation.png", dpi=200, bbox_inches="tight")
print("ok")
