#!/usr/bin/env python3
"""Three figures from traces_v2.npz, aligned by payload onset:
(A) controlled direct-vs-jailbreak (same behaviors); (B) success-vs-fail + direct;
(C) payload spike across templates 79/83/88."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("full_trait_output/persona_token_aggregate")
z = np.load(OUT / "traces_v2.npz", allow_pickle=True)
group, beh, onset, traj = z["group"], z["behavior"], z["onset"], z["traj"]
W = 30
xx = np.arange(-W, W + 1)


def align(mask):
    segs = []
    for tr, on in zip(traj[mask], onset[mask]):
        seg = np.full(2 * W + 1, np.nan)
        for j in range(-W, W + 1):
            idx = on + j
            if 0 <= idx < len(tr):
                seg[j + W] = tr[idx]
        segs.append(seg)
    M = np.array(segs); n = np.sum(~np.isnan(M), axis=0)
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1)), n


def line(ax, mask, col, lab):
    m, se, n = align(mask); keep = n >= 5
    ax.plot(xx[keep], m[keep], color=col, lw=2.4, label=f"{lab} (n={int(mask.sum())})")
    ax.fill_between(xx[keep], (m - 1.96 * se)[keep], (m + 1.96 * se)[keep], alpha=0.2, color=col)


def finish(ax, title, fn):
    ax.axvline(0, color="k", ls="--", lw=1.3, label="payload onset")
    ax.axhline(0, color="red", ls=":", lw=1.1)
    ax.set_xlabel("token position relative to payload onset")
    ax.set_ylabel("3-trait compliance logit (+ comply / - refuse)")
    ax.set_title(title); ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    ax.figure.tight_layout(); ax.figure.savefig(OUT / fn, dpi=180); print("saved", fn)


# B: success vs fail + direct
fig, ax = plt.subplots(figsize=(10, 5))
line(ax, group == "direct", "#3b6fb0", "direct request (refused)")
line(ax, group == "t79_succ", "#1a9641", "template-79 success")
line(ax, group == "t79_fail", "#c0392b", "template-79 fail")
finish(ax, "Compliance signal over the payload: direct vs successful/failed jailbreak", "fig_success_fail_direct.png")

# A: controlled — SAME behaviors, direct vs t79 jailbreak
flipset = set(beh[group == "t79_succ"]) & set(beh[group == "direct"])
inflip = np.array([b in flipset for b in beh])
fig, ax = plt.subplots(figsize=(10, 5))
line(ax, (group == "direct") & inflip, "#3b6fb0", "direct (refused)")
line(ax, (group == "t79_succ") & inflip, "#1a9641", "template-79 jailbreak (complied)")
finish(ax, f"Controlled: SAME {len(flipset)} behaviors, direct vs jailbreak (payload held constant)", "fig_controlled.png")

# C: multi-template successes
fig, ax = plt.subplots(figsize=(10, 5))
line(ax, group == "t79_succ", "#1a9641", "template 79")
line(ax, group == "t83_succ", "#e08a1e", "template 83")
line(ax, group == "t88_succ", "#7d3ac1", "template 88")
finish(ax, "Payload spike across three jailbreak templates (successful)", "fig_multitemplate.png")

for g in ["direct", "t79_succ", "t79_fail", "t83_succ", "t88_succ"]:
    m, _, _ = align(group == g)
    print(f"{g}: payload mean(x>=2) = {np.nanmean(m[xx >= 2]):+.2f}")
