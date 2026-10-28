#!/usr/bin/env python3
"""
Why does the jailbreak direction (dmu) not lose AUC when its persona component is removed?
For each direction w (intact, persona component only, persona removed) on HarmBench human_jailbreak (all rows):
  signal  = mean(h.w | jailbroken) - mean(h.w | refused)
  noise   = pooled within-class standard deviation of h.w
  d'      = signal / noise, and folded AUC
  noise ratio = within-class std along w / median within-class std over 500 random directions
Then a non-persona control for "removing high-variance directions": remove from dmu / refusal the top-k principal
components of the HarmBench activations (no labels), k = 10, 50, 229, and recompute AUC (HarmBench and the 3
same-orientation families GCG / PAIR / PEZ). Saved activations only. Writes why_dmu_no_drop.json.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
from llama_50seed_span import load_direction  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
fold = lambda a: max(a, 1 - a)
cfg = S.CFG["llama"]
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True); Qp = np.linalg.qr(T.T)[0]
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16); Xh = Xh.astype(np.float64)
fam = {}
for f in ("GCG", "PAIR", "PEZ"):
    rp, ap = cfg["fams"][f]; X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16); fam[f] = (X.astype(np.float64), y)
ref = unit(-load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt").astype(np.float64))
dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))
within = lambda w: np.sqrt(((Xh[yh == 1] @ w).var() + (Xh[yh == 0] @ w).var()) / 2)
rng = np.random.default_rng(0)
base = np.median([within(unit(rng.standard_normal(4096))) for _ in range(500)])
xf3 = lambda w: float(np.mean([fold(roc_auc_score(y, X @ w)) for X, y in fam.values()]))
res = {}
print(f"{'direction':28s} {'|signal|':>8s} {'noise':>7s} {'noise/random':>12s} {'d-prime':>7s} {'HB AUC':>7s} {'3-fam xfer':>10s}")
for n, v in (("jailbreak (dmu)", dmu), ("refusal", ref)):
    pin = Qp @ (Qp.T @ v)
    for tag, w in (("intact", v), ("persona component only", unit(pin)), ("persona removed", unit(v - pin))):
        s = (Xh[yh == 1] @ w).mean() - (Xh[yh == 0] @ w).mean(); nz = within(w)
        r = dict(signal=float(s), noise=float(nz), noise_ratio=float(nz / base), dprime=float(s / nz),
                 hb=fold(roc_auc_score(yh, Xh @ w)), xf3=xf3(w)); res[f"{n}|{tag}"] = r
        print(f"{n + ', ' + tag:28s} {abs(s):8.3f} {nz:7.3f} {nz / base:12.2f} {abs(s) / nz:7.3f} {r['hb']:7.3f} {r['xf3']:10.3f}")
print(f"(median within-class std along a random direction: {base:.3f})")
U = np.linalg.svd(Xh - Xh.mean(0), full_matrices=False)[2]
print("\nremove the top-k HarmBench principal components instead (no persona, no labels)")
for n, v in (("jailbreak (dmu)", dmu), ("refusal", ref)):
    for k in (0, 10, 50, 229):
        Qk = U[:k].T; w = unit(v - Qk @ (Qk.T @ v)) if k else v
        sh = float(np.sum((Qk.T @ v) ** 2)) if k else 0.0
        r = dict(share_removed=sh, hb=fold(roc_auc_score(yh, Xh @ w)), xf3=xf3(w), noise_ratio=float(within(w) / base)); res[f"{n}|pc{k}"] = r
        print(f"  {n:16s} k={k:3d}: share removed {sh:.3f}  noise/random {r['noise_ratio']:.2f}  HB {r['hb']:.3f}  3-fam xfer {r['xf3']:.3f}")
json.dump(res, open(HERE / "why_dmu_no_drop.json", "w"), indent=1)

# how much of the top-10 HarmBench PC subspace lies in persona space, and where the persona-removed dmu lives
sh_pc = np.sum((Qp.T @ U[:10].T) ** 2, 0)
w = unit(dmu - Qp @ (Qp.T @ dmu)); pin = unit(Qp @ (Qp.T @ dmu))
print("\nshare of each of the top-10 HarmBench PCs inside persona space:", " ".join(f"{s:.2f}" for s in sh_pc), f"(mean {sh_pc.mean():.2f})")
print(f"share of persona-removed dmu inside the top-10 PCs: {np.sum((U[:10] @ w) ** 2):.3f}; of its persona component: {np.sum((U[:10] @ pin) ** 2):.3f}")
for tag, v in (("persona component", pin), ("persona removed", w)):
    a = [fold(roc_auc_score(yh, Xh @ U[i])) for i in range(10)]
    c = (U[:10] @ v) ** 2
    print(f"  {tag:18s} weight on PC1..10: " + " ".join(f"{x:.2f}" for x in c))
print("  HarmBench AUC of PC1..10 alone:      " + " ".join(f"{x:.2f}" for x in a))
res["top10_pc_share_in_persona"] = sh_pc.tolist()
json.dump(res, open(HERE / "why_dmu_no_drop.json", "w"), indent=1)
