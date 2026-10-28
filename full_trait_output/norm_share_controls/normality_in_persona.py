#!/usr/bin/env python3
"""
Is "normality" (the benign-vs-harmful request axis) a region of persona space, and is normal = low on traits?
No jailbreak labels are used to build anything; HarmBench / transfer labels are only for scoring. Folded AUC.

Normality data: synthetic_v2 plain prompts from one activation run (roleplay_check):
  benign  = alpaca plain + jbb_benign plain;  harmful = advbench + strongreject + jbb plain.
T1 per-trait signs: effect size (mean diff / pooled sd) of benign - harmful, and of jailbroken - refused on
   HarmBench; fraction of traits that are lower for benign / jailbroken; correlation of the two across traits.
   Same for Gaussian directions (signs carry no meaning there, so ~50% is the null for the fractions).
T2 normality direction n = mean(benign) - mean(harmful): share of ||n||^2 in each 229-dim span, and AUC of h.n.
T3 normality detector: StandardScaler + LR C=10 trained benign vs harmful on the 229 projections (no jailbreak
   labels), scored on HarmBench human_jailbreak and the 5 transfer families. Bases: traits, Wikipedia, No Robots,
   Gaussian (5 draws each for controls), plus the full 4096-dim activation as reference.
T4 "persona intensity": mean of trait projections (z-scored on benign), as a single score.
Writes normality_in_persona.json.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(HERE))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402
fold = lambda a: np.maximum(a, 1 - a)
unit = lambda v: v / np.linalg.norm(v)
auc1 = lambda y, s: float(fold(rank_auc(y, np.asarray(s, np.float64)[:, None]))[0])
sauc1 = lambda y, s: float(rank_auc(y, np.asarray(s, np.float64)[:, None])[0])

cfg = S.CFG["llama"]
names = json.load(open(OUT / "trait_names_layer16.json"))
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
cb, eb = np.load(HERE / "control_bases_llama.npz"), np.load(HERE / "extra_bases_llama.npz")
rows = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
v2 = torch.load(OUT / "roleplay_check/synthetic_v2_acts.pt", map_location="cpu")
pl = [r for r in rows if r["frame"] == "plain"]
Xn = np.stack([v2[r["id"]].float().numpy() for r in pl]).astype(np.float64)
yn = np.array([1 if r["source"] in ("alpaca", "jbb_benign") else 0 for r in pl])  # 1 = benign
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
Xh = Xh.astype(np.float64)
fam = {}
for f, (rp, ap) in cfg["fams"].items():
    X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)
    fam[f] = (X.astype(np.float64), y)
print(f"normality set: {yn.sum()} benign, {(1 - yn).sum()} harmful plain prompts; HarmBench {len(yh)} ({yh.sum()} jailbroken)")
res = {}

def eff(Z, y):  # standardized mean difference, class 1 minus class 0, per column
    a, b = Z[y == 1], Z[y == 0]
    sd = np.sqrt((a.var(0) + b.var(0)) / 2)
    return (a.mean(0) - b.mean(0)) / sd

# T1
print("\nT1  per-direction sign of the effect (benign - harmful plain; jailbroken - refused HarmBench)")
print(f"{'dictionary':22s} {'% lower for benign':>19s} {'% lower for jailbroken':>23s} {'corr across dirs':>17s}")
for n, V in [("Persona traits", T)] + [(f"Gaussian d{i}", cb[f"gauss_d{i}"].astype(np.float64)) for i in range(2)]:
    en, ej = eff(Xn @ V.T, yn), eff(Xh @ V.T, yh)
    c = float(np.corrcoef(en, ej)[0, 1])
    print(f"{n:22s} {100 * np.mean(en < 0):18.0f}% {100 * np.mean(ej < 0):22.0f}% {c:17.2f}")
    if n == "Persona traits":
        res["T1"] = {"trait": names, "effect_benign_minus_harmful": en.tolist(), "effect_jb_minus_refused": ej.tolist(), "corr": c}
        o = np.argsort(en)
        print("   traits most LOWER for benign:  " + ", ".join(f"{names[i]} {en[i]:+.2f}/{ej[i]:+.2f}" for i in o[:8]))
        print("   traits most HIGHER for benign: " + ", ".join(f"{names[i]} {en[i]:+.2f}/{ej[i]:+.2f}" for i in o[::-1][:8]))
        print("   (pairs = effect benign-harmful / effect jailbroken-refused)")

# T2
nd = unit(Xn[yn == 1].mean(0) - Xn[yn == 0].mean(0))
def share(V, d):
    Q, _ = np.linalg.qr(V.T); return float(np.sum((Q.T @ d) ** 2))
print(f"\nT2  normality direction n: HarmBench AUC of h.n = {auc1(yh, Xh @ nd):.3f} "
      f"(signed {sauc1(yh, Xh @ nd):.3f}, >0.5 = jailbroken more benign-like); transfer mean "
      f"{np.mean([auc1(y, X @ nd) for X, y in fam.values()]):.3f}")
sh = {"Persona traits": share(T, nd), "Random Gaussian (5)": np.mean([share(cb[f'gauss_d{i}'], nd) for i in range(5)]),
      "Wikipedia acts (5)": np.mean([share(eb[f'wiki_d{i}'], nd) for i in range(5)]),
      "No Robots acts (5)": np.mean([share(eb[f'nr_d{i}'], nd) for i in range(5)])}
print("    share of ||n||^2 inside each 229-dim span: " + ", ".join(f"{k} {v:.3f}" for k, v in sh.items()))
res["T2"] = {"auc_hb": auc1(yh, Xh @ nd), "share": sh}

# T3
def norm_det(V):
    if V is None: m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=4000)).fit(Xn, yn); P = lambda X: X
    else: m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=4000)).fit(Xn @ V.T, yn); P = lambda X: X @ V.T
    s = m.decision_function(P(Xh))
    return auc1(yh, s), sauc1(yh, s), float(np.mean([auc1(y, m.decision_function(P(X))) for X, y in fam.values()]))
print("\nT3  detector trained ONLY benign-vs-harmful (no jailbreak labels), scored on jailbreak success")
print(f"{'basis':24s} {'HarmBench':>9s} {'(signed)':>9s} {'transfer':>9s}")
T3 = {}
for n, Vs in [("Persona traits", [T]), ("Wikipedia acts (5)", [eb[f"wiki_d{i}"] for i in range(5)]),
              ("No Robots acts (5)", [eb[f"nr_d{i}"] for i in range(5)]), ("Random Gaussian (5)", [cb[f"gauss_d{i}"] for i in range(5)]),
              ("Full 4096 activation", [None])]:
    r = np.array([norm_det(None if V is None else V.astype(np.float64)) for V in Vs]); T3[n] = r.tolist()
    print(f"{n:24s} {r[:, 0].mean():9.3f} {r[:, 1].mean():9.3f} {r[:, 2].mean():9.3f}")
res["T3"] = T3

# T4
mu, sd = (Xn[yn == 1] @ T.T).mean(0), (Xn[yn == 1] @ T.T).std(0)
inten = lambda X: (((X @ T.T) - mu) / sd).mean(1)
print(f"\nT4  persona intensity (mean z-scored trait projection): benign {inten(Xn[yn == 1]).mean():+.2f}, "
      f"harmful {inten(Xn[yn == 0]).mean():+.2f}; HarmBench jailbroken {inten(Xh[yh == 1]).mean():+.2f}, refused {inten(Xh[yh == 0]).mean():+.2f}; "
      f"HarmBench AUC {auc1(yh, inten(Xh)):.3f}")
res["T4"] = {"auc_hb": auc1(yh, inten(Xh))}
json.dump(res, open(HERE / "normality_in_persona.json", "w"))
