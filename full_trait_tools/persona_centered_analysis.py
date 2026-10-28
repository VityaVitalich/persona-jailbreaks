#!/usr/bin/env python3
"""Offline (CPU): per-token persona projections + a classifier trained on ONLY the
plotted traits (same features as the persona lines). The 3-trait logreg is trained on
HarmBench; the black line is its per-token logit, ending at the readout verdict.
From saved user_acts.npz (full-sequence activations + user span)."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np, torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("full_trait_output/persona_token_trace")
VECS = "full_trait_output/paper_data/building_blocks/persona_trait_vectors_llama"
HB = "full_trait_output/paper_data/activations/harmbench"
PERSONAS = ["egalitarian", "utilitarian", "principled"]
L = 16

# trait vectors (unit) for the plotted personas
pv = {}
for p in PERSONAS:
    v = torch.load(f"{VECS}/{p}.pt", map_location="cpu", weights_only=False)["vector"].float().numpy()[L]
    pv[p] = v / (np.linalg.norm(v) + 1e-12)
V3 = np.stack([pv[p] for p in PERSONAS])  # [3, 4096]

# train a logreg on ONLY these 3 trait projections, on HarmBench human-jailbreak pairs
acts_hb = torch.load(f"{HB}/activations.pt", map_location="cpu", weights_only=False)
Xtr, ytr = [], []
for l in open(f"{HB}/classified_responses.jsonl"):
    if not l.strip():
        continue
    r = json.loads(l)
    if r.get("attack_type") != "human_jailbreak" or r.get("jailbroken") is None:
        continue
    pid = r["pair_id"]; key = pid if pid in acts_hb else str(pid)
    if key not in acts_hb:
        continue
    Xtr.append(acts_hb[key]["16"].float().numpy() @ V3.T); ytr.append(int(bool(r["jailbroken"])))
Xtr, ytr = np.array(Xtr), np.array(ytr)
clf3 = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=4000, C=1.0))])
clf3.fit(Xtr, ytr)
print(f"3-trait clf trained on {len(ytr)} HarmBench pairs, jb_rate={ytr.mean():.2f}")
del acts_hb


def logit3(A):  # [n,4096] -> per-row 3-trait logit (positive = comply/jailbroken)
    return clf3.decision_function(np.atleast_2d(A) @ V3.T)


z = np.load(D / "user_acts.npz", allow_pickle=True)
names = list(z["names"])
full = [z[f"acts_{i}"] for i in range(len(names))]
span = [tuple(int(v) for v in z[f"span_{i}"]) for i in range(len(names))]
toks = [list(z[f"tokens_{i}"]) for i in range(len(names))]
comp = [bool(z[f"complied_{i}"]) for i in range(len(names))]
uacts = [full[i][span[i][0]:span[i][1]] for i in range(len(names))]
gmean = np.concatenate(uacts, axis=0).mean(axis=0)
rlim = max(abs(float(logit3(full[i][-1])[0])) for i in range(len(names))) * 1.2

groups = OrderedDict()
for i, n in enumerate(names):
    groups.setdefault(n.split(":")[0].strip(), []).append(i)


def draw_panel(ax, i):
    x = list(range(len(toks[i])))
    for p in PERSONAS:
        ax.plot(x, (uacts[i] - gmean) @ pv[p], marker="o", ms=3.5, lw=1.9, label=p)
    ax.axhline(0.0, color="k", ls=":", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([t.strip() for t in toks[i]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("persona projection\n(mean-subtracted)", fontsize=9); ax.grid(alpha=0.3)
    ax2 = ax.twinx(); ax2.set_ylim(-rlim, rlim)
    ltok = logit3(uacts[i]); lfin = float(logit3(full[i][-1])[0])
    ax2.plot(x + [len(x)], list(ltok) + [lfin], color="black", lw=1.8, marker="s", ms=2.5, alpha=0.85)
    ax2.axhline(0.0, color="red", ls="--", lw=1.4)
    comply = lfin > 0
    col = "#1a9641" if comply else "#c0392b"
    ax2.plot([len(x)], [lfin], marker="*", ms=22, color=col, markeredgecolor="k", zorder=6, clip_on=False)
    ax2.set_ylabel("3-trait classifier logit\n(+ comply / - refuse)", fontsize=8)
    disp = names[i].rsplit(" (", 1)[0]
    model = "COMPLIED" if comp[i] else "REFUSED"
    ax.set_title(f"{disp}    [model: {model}   |   classifier predicts: "
                 f"{'comply' if comply else 'refuse'} ({lfin:+.1f})]", loc="left", fontsize=10)


for key, idxs in groups.items():
    fig, axes = plt.subplots(len(idxs), 1, figsize=(12, 3.1 * len(idxs)), squeeze=False)
    for ax, i in zip(axes[:, 0], idxs):
        draw_panel(ax, i)
    axes[0, 0].plot([], [], color="black", lw=1.8, label="3-trait classifier")
    axes[0, 0].plot([], [], color="red", ls="--", lw=1.4, label="classifier boundary")
    axes[0, 0].legend(fontsize=8, loc="upper left", ncol=5)
    fig.suptitle(f"Persona projection + 3-trait classifier across the sentence — {key} (layer {L})", fontsize=12)
    fig.text(0.5, 0.006, "Classifier trained on the same 3 trait projections shown (HarmBench) to predict "
             "compliance; compliance on a harmful prompt = jailbreak (benign compliance is out-of-distribution).",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    safe = key.replace("/", "-").replace(" ", "")
    fig.savefig(D / f"persona_trace_{safe}.png", dpi=180)
    print("saved", D / f"persona_trace_{safe}.png")
