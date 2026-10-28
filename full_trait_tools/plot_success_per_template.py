#!/usr/bin/env python3
"""Success-vs-fail per-token compliance signal for templates 79, 83, 88, aligned by
payload onset. y=0 is the classifier decision boundary (comply vs refuse)."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("full_trait_output/persona_token_aggregate")
z = np.load(OUT / "traces_v2.npz", allow_pickle=True)
group, onset, traj = z["group"], z["onset"], z["traj"]
W = 30
xx = np.arange(-W, W + 1)


def line(ax, mask, col, lab):
    if mask.sum() == 0:
        return
    segs = []
    for tr, on in zip(traj[mask], onset[mask]):
        seg = np.full(2 * W + 1, np.nan)
        for j in range(-W, W + 1):
            idx = on + j
            if 0 <= idx < len(tr):
                seg[j + W] = tr[idx]
        segs.append(seg)
    M = np.array(segs); n = np.sum(~np.isnan(M), axis=0)
    m = np.nanmean(M, axis=0); se = np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1))
    keep = n >= 5
    ax.plot(xx[keep], m[keep], color=col, lw=2.4, label=f"{lab} (n={int(mask.sum())})")
    ax.fill_between(xx[keep], (m - 1.96 * se)[keep], (m + 1.96 * se)[keep], alpha=0.2, color=col)
    return np.nanmean(m[xx >= 2])


fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, t in zip(axes, [13, 36, 56]):
    ms = line(ax, group == f"t{t}_succ", "#1a9641", "successful jailbreak")
    mf = line(ax, group == f"t{t}_fail", "#c0392b", "failed jailbreak")
    ax.axvline(0, color="k", ls="--", lw=1.3, label="payload onset")
    ax.axhline(0, color="red", ls=":", lw=1.4)
    ax.text(-W + 1, 0.15, "decision boundary (comply / refuse)", fontsize=8, color="red", va="bottom")
    ax.set_xlabel("token position relative to payload onset")
    ax.set_title(f"Template {t}   (payload mean: succ {ms:+.1f}, fail {mf:+.1f})", fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("3-trait compliance logit (+ comply / - refuse)")
axes[0].legend(loc="lower left", fontsize=9)
fig.suptitle("Per-token compliance signal, successful vs failed jailbreaks (3 templates)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "success_vs_fail_templates.png", dpi=180)
print("saved", OUT / "success_vs_fail_templates.png")
