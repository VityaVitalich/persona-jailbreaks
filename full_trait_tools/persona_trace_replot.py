#!/usr/bin/env python3
"""Replot persona-vector projections across the sentence from saved results.json (no GPU)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("full_trait_output/persona_token_trace")
d = json.load(open(D / "results.json"))
order = ["ATM-hacking (refused)", "rainbow-table (complied)", "bleach+vinegar (complied)"]
order = [o for o in order if o in d]
personas = list(next(iter(d.values()))["persona_proj"].keys())

fig, axes = plt.subplots(len(order), 1, figsize=(11, 2.7 * len(order)), squeeze=False)
for ax, name in zip(axes[:, 0], order):
    r = d[name]; x = list(range(len(r["tokens"])))
    for p in personas:
        ax.plot(x, r["persona_proj"][p], marker="o", ms=3.5, lw=1.8, label=p)
    ax.axhline(0.0, color="k", ls=":", lw=0.7)
    # faint classifier gauge on a twin axis
    ax2 = ax.twinx()
    ax2.plot(x, r["classifier_logit"], color="gray", ls="--", lw=1.1, alpha=0.55)
    ax2.set_ylabel("classifier logit", color="gray", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([t.strip() for t in r["tokens"]], rotation=45, ha="right", fontsize=8)
    tag = "JAILBREAK (complied)" if r["complied"] else "harmful request (refused)"
    ax.set_title(f"{name}  —  {tag}", loc="left", fontsize=10)
    ax.set_ylabel("persona projection\n(cosine)", fontsize=9)
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=9, loc="upper left", ncol=len(personas))
fig.suptitle("Persona-vector projection across the sentence (layer 16)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(D / "persona_projection_trace.png", dpi=180)
print("saved", D / "persona_projection_trace.png")
