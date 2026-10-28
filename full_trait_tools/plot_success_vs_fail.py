#!/usr/bin/env python3
"""Two mean per-token logit curves: successful vs failed template-79 jailbreaks."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("full_trait_output/persona_token_aggregate")
z = np.load(OUT / "success_vs_fail.npz", allow_pickle=True)


def agg(trajs):
    Lmax = max(len(t) for t in trajs)
    M = np.full((len(trajs), Lmax), np.nan)
    for i, t in enumerate(trajs):
        M[i, :len(t)] = t
    n = np.sum(~np.isnan(M), axis=0)
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1)), n


sm, sse, sn = agg(z["succ_traj"])
fm, fse, fn = agg(z["fail_traj"])
# payload onset: scaffolding is identical -> within-group std ~0 until payload begins
Msucc = z["succ_traj"]
Lmax = max(len(t) for t in Msucc)
S = np.full((len(Msucc), Lmax), np.nan)
for i, t in enumerate(Msucc):
    S[i, :len(t)] = t
onset = int(np.argmax(np.nanstd(S, axis=0) > 0.05))

s = 2
fig, ax = plt.subplots(figsize=(11, 5))
for m, se, n, col, lab in [(sm, sse, sn, "#1a9641", f"successful jailbreak (n={len(z['succ_traj'])})"),
                           (fm, fse, fn, "#c0392b", f"failed jailbreak (n={len(z['fail_traj'])})")]:
    keep = n >= max(5, n.max() // 3)
    xx = np.arange(len(m))
    ax.plot(xx[s:][keep[s:]], m[s:][keep[s:]], color=col, lw=2.4, label=lab)
    ax.fill_between(xx[s:][keep[s:]], (m - 1.96 * se)[s:][keep[s:]], (m + 1.96 * se)[s:][keep[s:]], alpha=0.2, color=col)
ax.axvline(onset, color="k", ls="--", lw=1.4, label=f"payload onset (pos {onset})")
ax.axhline(0, color="red", ls=":", lw=1.2)
ax.axvspan(s, onset, color="tab:blue", alpha=0.06)
ax.text((s + onset) / 2, ax.get_ylim()[1], "identical jailbreak scaffolding", ha="center", va="top",
        fontsize=9, color="tab:blue")
ax.set_xlabel("token position")
ax.set_ylabel("3-trait compliance logit (+ comply / - refuse)")
ax.set_title("Per-token compliance signal: successful vs failed template-79 jailbreaks")
ax.legend(loc="lower left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "success_vs_fail.png", dpi=180)
print("saved", OUT / "success_vs_fail.png")
print(f"onset {onset}; payload mean: succ {np.nanmean(sm[onset:]):+.2f}, fail {np.nanmean(fm[onset:]):+.2f}")
