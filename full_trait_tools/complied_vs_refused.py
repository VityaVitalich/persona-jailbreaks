#!/usr/bin/env python3
"""(left) 3-trait compliance-signal logit for complied vs refused HarmBench prompts
(held-out split, with AUC); (right) verdict flip direct->jailbreak from aggregate.npz."""
import json
from pathlib import Path
import numpy as np, torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

VECS = "full_trait_output/paper_data/building_blocks/persona_trait_vectors_llama"
HB = "full_trait_output/paper_data/activations/harmbench"
AGG = Path("full_trait_output/persona_token_aggregate")
PERSONAS = ["egalitarian", "utilitarian", "principled"]
L = 16

pv = []
for p in PERSONAS:
    v = torch.load(f"{VECS}/{p}.pt", map_location="cpu", weights_only=False)["vector"].float().numpy()[L]
    pv.append(v / np.linalg.norm(v))
V3 = np.stack(pv)

acts = torch.load(f"{HB}/activations.pt", map_location="cpu", weights_only=False)
pairs = []
for l in open(f"{HB}/classified_responses.jsonl"):
    if not l.strip():
        continue
    r = json.loads(l)
    if r.get("attack_type") != "human_jailbreak" or r.get("jailbroken") is None:
        continue
    k = r["pair_id"] if r["pair_id"] in acts else str(r["pair_id"])
    if k not in acts:
        continue
    pairs.append((r["behavior_id"], acts[k]["16"].float().numpy() @ V3.T, int(bool(r["jailbroken"]))))
del acts

behs = sorted({b for b, _, _ in pairs})
rng = np.random.default_rng(41); rng.shuffle(behs)
trainb = set(behs[:int(0.7 * len(behs))])
Xtr = np.array([f for b, f, y in pairs if b in trainb]); ytr = np.array([y for b, f, y in pairs if b in trainb])
Xte = np.array([f for b, f, y in pairs if b not in trainb]); yte = np.array([y for b, f, y in pairs if b not in trainb])
clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=4000))]).fit(Xtr, ytr)
lte = clf.decision_function(Xte)
auc = roc_auc_score(yte, lte)
refused, complied = lte[yte == 0], lte[yte == 1]
print(f"held-out n={len(yte)} (refused {len(refused)}, complied {len(complied)}); AUC={auc:.3f}")

z = np.load(AGG / "aggregate.npz", allow_pickle=True)
dr, jr = z["dr_read"], z["jb_read"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
rng2 = np.random.default_rng(0)
for xpos, vals, col, lab in [(0, refused, "#c0392b", "refused"), (1, complied, "#1a9641", "complied")]:
    a1.scatter(xpos + rng2.uniform(-0.12, 0.12, len(vals)), vals, s=10, alpha=0.35, color=col)
    a1.plot([xpos - 0.2, xpos + 0.2], [vals.mean()] * 2, color=col, lw=3, zorder=5)
a1.axhline(0, color="red", ls="--", lw=1.2)
a1.set_xticks([0, 1]); a1.set_xticklabels(["model\nrefused", "model\ncomplied"]); a1.set_xlim(-0.5, 1.5)
a1.set_ylabel("3-trait compliance logit (+ comply / - refuse)")
a1.set_title(f"3-trait signal separates outcomes (held-out HarmBench, n={len(yte)}, AUC={auc:.2f})", fontsize=10)
a1.grid(alpha=0.3)

for d, j in zip(dr, jr):
    a2.plot([0, 1], [d, j], color=("gray" if j > d else "#d19a9a"), alpha=0.5, lw=0.8, zorder=1)
a2.scatter(np.zeros(len(dr)), dr, color="#c0392b", s=28, zorder=3)
a2.scatter(np.ones(len(jr)), jr, color="#1a9641", s=28, zorder=3)
a2.plot([-0.13, 0.13], [dr.mean()] * 2, color="#c0392b", lw=3)
a2.plot([0.87, 1.13], [jr.mean()] * 2, color="#1a9641", lw=3)
a2.axhline(0, color="red", ls="--", lw=1.2)
a2.set_xticks([0, 1]); a2.set_xticklabels(["direct\nrequest", "template-79\njailbreak"]); a2.set_xlim(-0.5, 1.5)
a2.set_ylabel("3-trait compliance logit at readout")
a2.set_title(f"Jailbreak flips the verdict (n={len(dr)}, {(jr>dr).sum()}/{len(dr)} up)\n"
             f"direct {dr.mean():+.1f} -> jailbreak {jr.mean():+.1f}", fontsize=10)
a2.grid(alpha=0.3)
fig.suptitle("3-trait compliance signal: separates complied/refused, and the jailbreak flips it", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(AGG / "complied_vs_refused.png", dpi=180)
print("saved", AGG / "complied_vs_refused.png")
