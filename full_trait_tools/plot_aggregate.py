#!/usr/bin/env python3
"""Aggregate plot. Template-79 scaffolding is identical across prompts, so trajectories
align at the start; the payload begins where they diverge (cross-prompt std > 0).
(left) mean 3-trait logit vs position, payload onset marked, CI over the payload region;
(right) paired direct-vs-jailbreak readout verdict."""
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("full_trait_output/persona_token_aggregate")
z = np.load(OUT / "aggregate.npz", allow_pickle=True)
jb_traj = z["jb_traj"]; dr, jr = z["dr_read"], z["jb_read"]
pval = float(wilcoxon(jr, dr).pvalue)

Lmax = max(len(t) for t in jb_traj)
M = np.full((len(jb_traj), Lmax), np.nan)
for i, t in enumerate(jb_traj):
    M[i, :len(t)] = t
mean = np.nanmean(M, axis=0)
nn = np.sum(~np.isnan(M), axis=0)
std = np.nanstd(M, axis=0)
se = std / np.sqrt(np.maximum(nn, 1))
# payload onset = first position where the trajectories diverge (identical scaffolding => std~0)
onset = int(np.argmax(std > 0.05)) if np.any(std > 0.05) else 0
# only keep positions with enough prompts
keep = nn >= max(5, len(jb_traj) // 3)
x = np.arange(Lmax)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
# left: clip the first-token artifact (start at pos 2) and focus the scale
s = 2
a1.plot(x[s:][keep[s:]], mean[s:][keep[s:]], color="black", lw=2.2)
a1.fill_between(x[s:][keep[s:]], (mean - 1.96 * se)[s:][keep[s:]], (mean + 1.96 * se)[s:][keep[s:]],
                alpha=0.25, color="gray")
a1.axvline(onset, color="red", ls="--", lw=1.5, label=f"payload onset (pos {onset})")
a1.axhline(0, color="k", ls=":", lw=0.8)
a1.axvspan(s, onset, color="tab:blue", alpha=0.06)
vis = mean[s:][keep[s:]]
a1.set_ylim(min(vis.min(), -1) - 1, vis.max() + 2)
a1.text((s + onset) / 2, a1.get_ylim()[1], "jailbreak scaffolding", ha="center", va="top", fontsize=9, color="tab:blue")
a1.set_xlabel("token position")
a1.set_ylabel("3-trait classifier logit (+ comply / - refuse)")
a1.set_title(f"Mean logit across the sentence ({len(jb_traj)} template-79 jailbreaks)")
a1.legend(loc="lower right"); a1.grid(alpha=0.3)

order = np.argsort(dr)
a2.scatter(np.zeros(len(dr)), dr, color="#c0392b", s=28, zorder=3)
a2.scatter(np.ones(len(jr)), jr, color="#1a9641", s=28, zorder=3)
for d, j in zip(dr, jr):
    a2.plot([0, 1], [d, j], color=("gray" if j > d else "#d19a9a"), alpha=0.5, lw=0.8, zorder=1)
a2.plot([-0.13, 0.13], [dr.mean()] * 2, color="#c0392b", lw=3)
a2.plot([0.87, 1.13], [jr.mean()] * 2, color="#1a9641", lw=3)
a2.axhline(0, color="red", ls="--", lw=1.2)
a2.set_xticks([0, 1]); a2.set_xticklabels(["direct\nrequest", "template-79\njailbreak"])
a2.set_xlim(-0.4, 1.4)
a2.set_ylabel("3-trait logit at readout (+ comply / - refuse)")
pv = "p<0.001" if pval < 1e-3 else f"p={pval:.3f}"
a2.set_title(f"Readout verdict flips (n={len(dr)}, {(jr>dr).sum()}/{len(dr)} up, Wilcoxon {pv})\n"
             f"direct {dr.mean():+.1f} -> jailbreak {jr.mean():+.1f}", fontsize=10)
a2.grid(alpha=0.3)
fig.suptitle("Aggregate: template-79 jailbreak flips the 3-trait compliance verdict", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / "aggregate.png", dpi=180)
print("saved", OUT / "aggregate.png")
print(f"onset pos {onset}; scaffolding mean logit {np.nanmean(mean[:onset]):+.2f}; "
      f"payload mean logit {np.nanmean(mean[onset:][keep[onset:]]):+.2f}")
print(f"readout: direct {dr.mean():+.2f}+-{dr.std():.2f}, jailbreak {jr.mean():+.2f}+-{jr.std():.2f}; "
      f"jailbreak>direct in {(jr>dr).sum()}/{len(dr)}")
