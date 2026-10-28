# GEMMA PORT of norm_share_controls/what_is_dmu_out.py (gemma/port.py); model-specific inputs swapped, logic unchanged.
#!/usr/bin/env python3
"""
What does the jailbreak direction (dmu) carry outside persona space?  dmu = mean(jb) - mean(refused) on HarmBench
human_jailbreak (all rows); w_in = its persona component, w_out = dmu minus w_in (both unit). Saved activations only.
  1. variance of the score h.w across HarmBench prompts explained by the jailbreak template, by the behaviour
     (request), and by both (additive fixed effects, R^2)
  2. separation of jailbroken from refused: overall, within a template (mean over templates), within a behaviour
     (mean over behaviours); reference: the template's jailbreak rate and the behaviour's jailbreak rate as scores
  3. on the plain / role-play requests of roleplay_check synthetic_v2: harmful vs benign plain requests, and
     role-play framing vs plain on the same requests
  4. cosine with the refusal direction and with the normality direction (mean benign - mean harmful plain)
  5. the templates and behaviours at the two ends of h.w_out
All AUCs folded. Writes what_is_dmu_out.json.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; OUT = HERE.parent.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
sys.path.insert(0, "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma"); from gemma_common import _gemma_arditi  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
fold = lambda a: max(a, 1 - a)
cfg = S.CFG["gemma"]
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True); Qp = np.linalg.qr(T.T)[0]
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, rows = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 30); Xh = Xh.astype(np.float64)
tpl = np.array([str(r["jailbreak_idx"]) for r in rows]); beh = np.array([r["behavior_id"] for r in rows])
ref = unit(-_gemma_arditi().astype(np.float64))
dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0)); pin = Qp @ (Qp.T @ dmu)
W = {"dmu intact": dmu, "persona component (w_in)": unit(pin), "persona removed (w_out)": unit(dmu - pin)}
v2rows = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
v2 = torch.load(Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/synthetic_v2_acts.pt"), map_location="cpu")
Xv = np.stack([v2[r["id"]].float().numpy() for r in v2rows]).astype(np.float64)
plain = np.array([r["frame"] == "plain" for r in v2rows]); harm = np.array([r["harm"] == "harmful" for r in v2rows])
Xn = Xv[plain]; nd = unit(Xn[~harm[plain]].mean(0) - Xn[harm[plain]].mean(0))

def onehot(lab):
    u, inv = np.unique(lab, return_inverse=True); M = np.zeros((len(lab), len(u))); M[np.arange(len(lab)), inv] = 1; return M
def r2(s, *labs):
    M = np.hstack([onehot(l) for l in labs]); c, *_ = np.linalg.lstsq(M, s - s.mean(), rcond=None)
    return 1 - np.var(s - s.mean() - M @ c) / np.var(s)
def within_auc(s, groups):
    a, n = [], []
    for g in np.unique(groups):
        m = groups == g
        if 0 < yh[m].sum() < m.sum(): a.append(roc_auc_score(yh[m], s[m])); n.append(m.sum())
    return float(np.average([fold(x) for x in a], weights=n))
rate = lambda groups: np.array([yh[groups == g].mean() for g in groups])
res = {}
print(f"reference: template jailbreak rate as score AUC {fold(roc_auc_score(yh, rate(tpl))):.3f}; behaviour rate {fold(roc_auc_score(yh, rate(beh))):.3f}; "
      f"label variance explained by template {r2(yh.astype(float), tpl):.2f}, by behaviour {r2(yh.astype(float), beh):.2f}")
print(f"\n{'direction':26s} | R2 template  behaviour  both | AUC all  within-template  within-behaviour | harmful vs benign  role-play vs plain | cos refusal  cos normality")
for n, w in W.items():
    s = Xh @ w; sv = Xv @ w
    r = dict(r2_template=r2(s, tpl), r2_behaviour=r2(s, beh), r2_both=r2(s, tpl, beh), auc=fold(roc_auc_score(yh, s)),
             auc_within_template=within_auc(s, tpl), auc_within_behaviour=within_auc(s, beh),
             harmful_vs_benign=fold(roc_auc_score(harm[plain], sv[plain])), roleplay_vs_plain=fold(roc_auc_score(~plain, sv)),
             cos_refusal=float(w @ ref), cos_normality=float(w @ nd))
    res[n] = r
    print(f"{n:26s} | {r['r2_template']:11.2f} {r['r2_behaviour']:10.2f} {r['r2_both']:5.2f} | {r['auc']:7.3f} {r['auc_within_template']:16.3f} {r['auc_within_behaviour']:17.3f} | "
          f"{r['harmful_vs_benign']:17.3f} {r['roleplay_vs_plain']:19.3f} | {r['cos_refusal']:+11.3f} {r['cos_normality']:+14.3f}")
s = Xh @ W["persona removed (w_out)"]
tm = sorted({(np.mean(s[tpl == t]), t, yh[tpl == t].mean(), (tpl == t).sum()) for t in np.unique(tpl)})
print("\ntemplates by mean h.w_out (low -> high): template id, mean score, jailbreak rate")
print("  " + "  ".join(f"[{t}] {m:+.2f} jb={j:.2f}" for m, t, j, _ in tm))
res["templates_by_w_out"] = [dict(template=t, mean=float(m), jb_rate=float(j), n=int(k)) for m, t, j, k in tm]
bm = sorted((np.mean(s[beh == b]), b) for b in np.unique(beh))
print("\nbehaviours lowest on h.w_out:  " + ", ".join(b for _, b in bm[:6]))
print("behaviours highest on h.w_out: " + ", ".join(b for _, b in bm[-6:]))
json.dump(res, open(HERE / "what_is_dmu_out.json", "w"), indent=1)

# baselines for the harmful-vs-benign and behaviour-severity readings
rngb = np.random.default_rng(0)
G = np.stack([unit(rngb.standard_normal(5376)) for _ in range(200)])
A = np.load(HERE / "control_bases_gemma.npz")["acts_d0"].astype(np.float64)
hv = lambda V: np.array([fold(roc_auc_score(harm[plain], Xn @ v)) for v in V])
for n, V in (("random Gaussian directions", G), ("random Alpaca activations", A)):
    a = hv(V); print(f"baseline harmful vs benign, {n}: median {np.median(a):.3f}, 95th pct {np.percentile(a, 95):.3f}")
    r2t = [r2(Xh @ v, tpl) for v in V[:50]]; print(f"   R2 template of h.v: median {np.median(r2t):.2f}")
res["baseline_harmful_vs_benign"] = {"gauss_median": float(np.median(hv(G))), "alpaca_median": float(np.median(hv(A)))}
# severity: behaviour mean of h.w_out vs behaviour jailbreak rate (Spearman)
from scipy.stats import spearmanr
ub = np.unique(beh); mb = np.array([s[beh == b].mean() for b in ub]); jb = np.array([yh[beh == b].mean() for b in ub])
print(f"behaviour-level Spearman(mean h.w_out, behaviour jailbreak rate) = {spearmanr(mb, jb)[0]:.2f} over {len(ub)} behaviours")
ut = np.unique(tpl); mt = np.array([s[tpl == t].mean() for t in ut]); jt = np.array([yh[tpl == t].mean() for t in ut])
print(f"template-level Spearman(mean h.w_out, template jailbreak rate) = {spearmanr(mt, jt)[0]:.2f} over {len(ut)} templates")
json.dump(res, open(HERE / "what_is_dmu_out.json", "w"), indent=1)
