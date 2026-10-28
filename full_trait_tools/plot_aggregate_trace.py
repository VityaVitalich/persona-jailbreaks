#!/usr/bin/env python3
"""Aggregate version of the single-example trace: mean persona-projection lines + mean
3-trait classifier line, aligned by payload onset, for direct (refused) vs template-79
jailbreak (complied). From traces_v2.npz (needs the 'persona' array)."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("full_trait_output/persona_token_aggregate")
z = np.load(OUT / "traces_v2.npz", allow_pickle=True)
group, onset, traj, persona = z["group"], z["onset"], z["traj"], z["persona"]
PERSONAS = ["egalitarian", "utilitarian", "principled"]
COLS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
W = 30
xx = np.arange(-W, W + 1)
gmean = np.concatenate(list(persona), axis=0).mean(axis=0)   # global mean per persona


def align(arrs, onsets):
    segs = []
    for a, on in zip(arrs, onsets):
        seg = np.full(2 * W + 1, np.nan)
        for j in range(-W, W + 1):
            idx = on + j
            if 0 <= idx < len(a):
                seg[j + W] = a[idx]
        segs.append(seg)
    M = np.array(segs); n = np.sum(~np.isnan(M), axis=0)
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1)), n


fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), sharey=True)
for ax, (g, lab) in zip(axes, [("direct", "direct request (refused)"),
                                ("jailbreak", "template-79 jailbreak (complied)")]):
    idx = np.where(group == g)[0]
    on = onset[idx]
    for k, (p, c) in enumerate(zip(PERSONAS, COLS)):
        m, se, n = align([persona[i][:, k] - gmean[k] for i in idx], on); keep = n >= 5
        ax.plot(xx[keep], m[keep], color=c, lw=2, label=p)
        ax.fill_between(xx[keep], (m - 1.96 * se)[keep], (m + 1.96 * se)[keep], alpha=0.15, color=c)
    ax.axhline(0, color="k", ls=":", lw=0.6)
    ax.axvline(0, color="k", ls="--", lw=1.2, label="payload onset")
    ax.set_xlabel("token position relative to payload onset")
    ax2 = ax.twinx(); ax2.set_ylim(-8, 8)
    m, se, n = align([traj[i] for i in idx], on); keep = n >= 5
    ax2.plot(xx[keep], m[keep], color="black", lw=2.2, label="3-trait classifier")
    ax2.fill_between(xx[keep], (m - 1.96 * se)[keep], (m + 1.96 * se)[keep], alpha=0.15, color="gray")
    ax2.axhline(0, color="red", ls="--", lw=1.3)
    ax2.set_ylabel("3-trait classifier logit\n(+ comply / - refuse)", fontsize=8)
    mr = np.mean([traj[i][-1] for i in idx])
    ax.set_title(f"{lab}   [mean verdict {mr:+.1f} = {'comply' if mr > 0 else 'refuse'}]", fontsize=11)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("persona projection (mean-subtracted)")
axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle(f"Aggregate persona + 3-trait classifier over the payload "
             f"(template-79, n={int((group == 'direct').sum())} behaviors, 95% CI)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "aggregate_trace.png", dpi=180)
print("saved", OUT / "aggregate_trace.png")
