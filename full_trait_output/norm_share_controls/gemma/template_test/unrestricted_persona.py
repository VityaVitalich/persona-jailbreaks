# GEMMA PORT of norm_share_controls/template_test/unrestricted_persona.py (gemma/port.py); model-specific inputs swapped, logic unchanged.
#!/usr/bin/env python3
"""
Is the part of the jailbreak direction outside the trait span (w_out) an "unrestricted AI" persona that the 240 trait
vectors do not capture?  No new personas: the direction is built from the existing role-play activations of
roleplay_check synthetic_v2 (layer 30, last token), where each of 1794 requests (897 harmful, 897 benign) is posed
under 2 jailbreak-persona framings (amoral_sage, dev_mode, evil_twin, opposite_day, rebel_ai, unbound), 2 neutral
personas (grandpa, pirate, teacher, ...), 2 neutral fictions and plainly.

  u_jbp   mean(jailbreak-persona framings) - mean(neutral-persona framings), averaged per request then over requests
          (role-play is present on both sides, so u_jbp is the "unrestricted AI" identity, not role-play itself)
  U6      span of the 6 per-framing directions mean(framing) - mean(neutral personas)
  null    the same construction with the neutral personas split into two random halves of framings (10 draws)
Tests
  (a) does w_out read the jailbreak persona?  within-request AUC of jailbreak-persona vs neutral-persona framings
      (synthetic set), for w_out, w_in, dmu, the refusal direction, a random Alpaca activation (median of 240)
  (b) is the jailbreak persona missing from the trait span?  share of u_jbp's squared norm in the trait span, vs
      the null contrast, vs a 240-dim span of Alpaca activations and a random span
  (c) is w_out that persona?  cos(w_out, u_jbp), share of w_out in U6; on HarmBench, u_jbp as a score (within-request
      AUC, rho_template), and w_out with U6 projected out
Writes unrestricted_persona.json.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; NS = HERE.parent; OUT = NS.parent.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
sys.path.insert(0, "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma"); from gemma_common import _gemma_arditi  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
fold = lambda a: max(a, 1 - a)
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]
cfg = S.CFG["gemma"]
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True); Qp = qr(T)
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, rows = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 30); Xh = Xh.astype(np.float64)
tpl = np.array([int(r["jailbreak_idx"]) for r in rows]); beh = np.array([r["behavior_id"] for r in rows])
ref = unit(-_gemma_arditi().astype(np.float64))
dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0)); pin = Qp @ (Qp.T @ dmu)
w_out, w_in = unit(dmu - pin), unit(pin)
cb = np.load(NS / "control_bases_gemma.npz")

R = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
A = torch.load(Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/synthetic_v2_acts.pt"), map_location="cpu")
X = np.stack([A[r["id"]].float().numpy() for r in R]).astype(np.float64)
req = np.array([f"{r['harm']}_{r['source']}_{r['req_idx']}" for r in R]); ft = np.array([r["frame_type"] for r in R]); fr = np.array([r["frame"] for r in R])
JB = sorted(set(fr[ft == "jb_persona"])); NP = sorted(set(fr[ft == "neutral_persona"]))

def contrast(pos_mask, neg_mask):  # per-request mean difference, averaged over requests that have both
    d, n = np.zeros(X.shape[1]), 0
    for q in np.unique(req[pos_mask]):
        a, b = pos_mask & (req == q), neg_mask & (req == q)
        if a.any() and b.any(): d += X[a].mean(0) - X[b].mean(0); n += 1
    return d / n
isj, isn = ft == "jb_persona", ft == "neutral_persona"
u_jbp = unit(contrast(isj, isn))
U6 = qr(np.stack([unit(contrast(fr == f, isn)) for f in JB]))
rng = np.random.default_rng(0)
nulls = []
for _ in range(10):
    p = rng.permutation(NP); h1, h2 = set(p[: len(p) // 2]), set(p[len(p) // 2:])
    nulls.append(unit(contrast(np.isin(fr, list(h1)), np.isin(fr, list(h2)))))
share = lambda Q, v: float(np.sum((Q.T @ v) ** 2))
res = {}

# (a) does w_out read the jailbreak persona? within-request AUC on the synthetic set
def within_syn(s):
    a = []
    for q in np.unique(req):
        m = (req == q) & (isj | isn)
        if isj[m].any() and isn[m].any(): a.append(roc_auc_score(isj[m], s[m]))
    return float(fold(np.mean(a)))
Aacts = cb["acts_d0"].astype(np.float64)
base = float(np.median([within_syn(X @ unit(v)) for v in Aacts[:100]]))
res["a"] = {n: within_syn(X @ v) for n, v in (("w_out", w_out), ("w_in", w_in), ("dmu", dmu), ("refusal", ref), ("u_jbp (in-sample)", u_jbp))}
res["a"]["random Alpaca activation (median of 100)"] = base
print("(a) jailbreak-persona vs neutral-persona framing of the same request, within-request AUC")
for k, v in res["a"].items(): print(f"    {k:42s} {v:.3f}")

# (b) is the jailbreak persona missing from the trait span?
QA = qr(cb["acts_d0"]); QG = qr(cb["gauss_d0"])
res["b"] = {"u_jbp": {"trait span": share(Qp, u_jbp), "Alpaca-activation span": share(QA, u_jbp), "random span": share(QG, u_jbp)},
            "null (neutral persona vs neutral persona), mean of 10": {"trait span": float(np.mean([share(Qp, v) for v in nulls])),
                                                                     "Alpaca-activation span": float(np.mean([share(QA, v) for v in nulls])),
                                                                     "random span": float(np.mean([share(QG, v) for v in nulls]))}}
print("\n(b) share of squared norm in each 240-dim span")
for k, v in res["b"].items(): print(f"    {k:52s} " + "  ".join(f"{a} {b:.3f}" for a, b in v.items()))

# (c) is w_out that persona?
def within_hb(s):
    a, n = [], []
    for u in np.unique(beh):
        m = beh == u
        if 0 < yh[m].sum() < m.sum(): a.append(fold(roc_auc_score(yh[m], s[m]))); n.append(m.sum())
    return float(np.average(a, weights=n))
jbt = {t: yh[tpl == t].mean() for t in np.unique(tpl)}
rho_t = lambda s: float(abs(spearmanr([s[tpl == t].mean() for t in jbt], list(jbt.values()))[0]))
u_out = unit(u_jbp - Qp @ (Qp.T @ u_jbp))
w_out_minus = unit(w_out - U6 @ (U6.T @ w_out))
res["c"] = {"cos(w_out, u_jbp)": float(w_out @ u_jbp), "cos(w_in, u_jbp)": float(w_in @ u_jbp),
            "share of w_out in U6 (6 dims)": share(U6, w_out), "share of a random direction in a 6-dim span": 6 / 5376}
print("\n(c) " + "  ".join(f"{k} {v:+.3f}" for k, v in res["c"].items()))
print(f"    {'HarmBench score':44s} {'within-request AUC':>18s} {'rho_template':>12s}")
for n, v in (("w_out", w_out), ("u_jbp", u_jbp), ("u_jbp, trait-span part removed", u_out), ("w_out with U6 projected out", w_out_minus)):
    s = Xh @ v; res["c"][f"hb|{n}"] = dict(within_request=within_hb(s), rho_template=rho_t(s))
    print(f"    {n:44s} {within_hb(s):18.3f} {rho_t(s):12.3f}")
json.dump(res, open(HERE / "unrestricted_persona.json", "w"), indent=1)

# (d) dropped in the Gemma port: it needs Gemma's own answered/refused labels for the synthetic prompts (generation + judge).
