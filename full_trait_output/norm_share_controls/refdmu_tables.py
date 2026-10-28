#!/usr/bin/env python3
"""
Numbers for the appendix tables on the refusal and jailbreak directions without their persona component.
Table A (per family): folded AUC of h.w, intact vs persona removed. HarmBench = held-out test set, mean over the 50
  splits (appendix_controls_data.json; dmu re-estimated on each split's train rows). Families: refusal as is, dmu
  estimated on all HarmBench rows (the paper's direction).
Table B (what each direction reads): for dmu intact / persona component / persona removed, refusal intact / persona
  removed, and the median over 229 random Alpaca activations and 229 random Gaussian directions:
  d' on HarmBench, separation within a template and within a request (folded AUC, weighted mean over groups),
  Spearman correlation of the per-request and per-template mean score with the jailbreak rate, cosine with the
  refusal direction. dmu is estimated on all HarmBench rows, so its within-group numbers are in-sample.
Writes refdmu_tables.json.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
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
Xh, yh, rows = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16); Xh = Xh.astype(np.float64)
tpl = np.array([str(r["jailbreak_idx"]) for r in rows]); beh = np.array([r["behavior_id"] for r in rows])
fam = {}
for f, (rp, ap) in cfg["fams"].items():
    X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16); fam[f] = (X.astype(np.float64), y)
ref = unit(-load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt").astype(np.float64))
dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))
rm = lambda v: unit(v - Qp @ (Qp.T @ v)); pc = lambda v: unit(Qp @ (Qp.T @ v))
D = json.load(open(HERE / "appendix_controls_data.json"))["auc"]
res = {"A": {}, "B": {}}
for n, v, key in (("refusal", ref, "refusal"), ("dmu", dmu, "dmu")):
    for tag, w, k in (("intact", v, "full"), ("persona removed", rm(v), "persona_removed")):
        r = {"HarmBench": D[key][k]["hb"]}
        for f, (X, y) in fam.items():
            a = roc_auc_score(y, X @ w); r[f] = fold(a); r[f + "_reversed"] = bool(a < 0.5)
        r["mean5"] = float(np.mean([r[f] for f in fam])); r["mean3"] = float(np.mean([r[f] for f in ("GCG", "PAIR", "PEZ")]))
        res["A"][f"{n}|{tag}"] = r
def within(s, g):
    a, n = [], []
    for u in np.unique(g):
        m = g == u
        if 0 < yh[m].sum() < m.sum(): a.append(fold(roc_auc_score(yh[m], s[m]))); n.append(m.sum())
    return float(np.average(a, weights=n))
def group_rho(s, g):
    u = np.unique(g); return float(abs(spearmanr([s[g == x].mean() for x in u], [yh[g == x].mean() for x in u])[0]))
def readout(w):
    s = Xh @ w; sig = s[yh == 1].mean() - s[yh == 0].mean(); nz = np.sqrt((s[yh == 1].var() + s[yh == 0].var()) / 2)
    return dict(dprime=float(abs(sig) / nz), within_template=within(s, tpl), within_request=within(s, beh),
                rho_request=group_rho(s, beh), rho_template=group_rho(s, tpl), cos_refusal=float(abs(w @ ref)))
for n, w in (("dmu intact", dmu), ("dmu persona component", pc(dmu)), ("dmu persona removed", rm(dmu)),
             ("refusal intact", ref), ("refusal persona removed", rm(ref))):
    res["B"][n] = readout(w)
rng = np.random.default_rng(0)
for n, V in (("random Alpaca activation (median of 229)", np.load(HERE / "control_bases_llama.npz")["acts_d0"].astype(np.float64)),
             ("random Gaussian direction (median of 229)", np.stack([unit(rng.standard_normal(4096)) for _ in range(229)]))):
    rs = [readout(unit(v)) for v in V]
    res["B"][n] = {k: float(np.median([r[k] for r in rs])) for k in rs[0]}
json.dump(res, open(HERE / "refdmu_tables.json", "w"), indent=1)
print("A"); [print(f"  {k:26s} " + " ".join(f"{f} {v[f]:.3f}{'r' if v.get(f + '_reversed') else ''}" for f in ["HarmBench", *fam]) + f"  mean5 {v['mean5']:.3f} mean3 {v['mean3']:.3f}") for k, v in res["A"].items()]
print("B"); [print(f"  {k:42s} " + " ".join(f"{a} {b:.3f}" for a, b in v.items())) for k, v in res["B"].items()]
