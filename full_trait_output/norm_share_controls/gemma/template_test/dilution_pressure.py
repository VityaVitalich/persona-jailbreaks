# GEMMA PORT of norm_share_controls/template_test/dilution_pressure.py (gemma/port.py); model-specific inputs swapped, logic unchanged.
#!/usr/bin/env python3
"""
Does the part of the jailbreak direction outside persona space (w_out) read (2) dilution of the request by the
template text or (3) compliance pressure written into the template?  HarmBench human_jailbreak, 3180 prompts =
159 requests x 20 templates; the request sits at the end of almost every template (harmbench_jailbreaks.py, the
HarmBench human_jailbreaks list the activations were built from).

Prompt features
  dilution   log #characters of template text before the request; log #characters after it; log request length
  pressure   keyword rules on the template text, one indicator each (the rules are printed with the labels):
             no_refusal, amoral, prefix_or_format, dual_response, ignore_rules, stay_in_character
Scores: w_out, w_in, intact dmu (all HarmBench rows; see what_is_dmu_out.py), each feature on its own.
Measures (folded AUC; the in-sample caveat of what_is_dmu_out.py applies to the dmu-based scores)
  within-request AUC  jailbroken vs refused among the templates applied to the same request (weighted mean)
  rho_template        |Spearman| of the per-template mean score with the template's jailbreak rate (20 templates)
  template oracle     leave-one-request-out template jailbreak rate as the score: all template-level information
Explaining w_out: regress w_out on the features (OLS over all prompts) and score the residual; if the features
explain what w_out reads, the residual's within-request AUC falls to the random-activation level (0.68,
refdmu_tables.json). Also the leave-one-template-out R^2 of predicting the per-template mean of w_out from the features.
Writes dilution_pressure.json.
"""
import json, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; NS = HERE.parent; OUT = NS.parent.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
fold = lambda a: max(a, 1 - a)
ns = {}; exec(open(HERE / "harmbench_jailbreaks.py").read(), ns); J = ns["JAILBREAKS"]
cfg = S.CFG["gemma"]
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True); Qp = np.linalg.qr(T.T)[0]
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, rows = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 30); Xh = Xh.astype(np.float64)
tpl = np.array([int(r["jailbreak_idx"]) for r in rows]); beh = np.array([r["behavior_id"] for r in rows])
dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0)); pin = Qp @ (Qp.T @ dmu)
SC = {"w_out": Xh @ unit(dmu - pin), "w_in": Xh @ unit(pin), "dmu": Xh @ dmu}

RULES = {
    "no_refusal": r"never refus|not refuse|n'?t refuse|no refus|without refus|cannot refuse|never (?:say|says) no|answers? any|any request|no restrictions|unfilt|uncensor|no (?:filter|limit|boundar)|bypass|can do anything|capable of anything",
    "amoral": r"amoral|unethical|immoral|no (?:morals|ethics)|n'?t care (?:about )?(?:any )?(?:laws|ethic|moral)|without (?:any )?(?:morals|ethics)",
    "prefix_or_format": r"start (?:your|each|every|all)\b[^.\n]{0,30}(?:response|reply|answer|output)|begin (?:your|each|every)|prefix|reply only with|respond only|always (?:start|begin)|in (?:the )?format|format:",
    "dual_response": r"two (?:different )?(?:separate )?(?:responses|ways|answers)|both (?:responses|answers)|\[gpt\]|one normal|normal (?:response|way|output)",
    "ignore_rules": r"ignore (?:all )?(?:the )?(?:previous|prior|instructions)|openai'?s? (?:content )?polic|content polic|guidelines",
    "stay_in_character": r"stay in character|stays in character|remain in character|break character|never break",
}
used = sorted(set(tpl))
lab = {t: {k: int(bool(re.search(p, J[t], re.I))) for k, p in RULES.items()} for t in used}
pre = {t: J[t].index("{0}") for t in used}
F = {"log_template_before": np.log1p([pre[t] for t in tpl]), "log_template_after": np.log1p([len(J[t]) - pre[t] - 3 for t in tpl]),
     "log_request_len": np.log1p([len(r["behavior_text"]) for r in rows])}
for k in RULES: F[k] = np.array([lab[t][k] for t in tpl], float)
jbt = {t: yh[tpl == t].mean() for t in used}
print("template labels (keyword rules) and jailbreak rate:")
print(f"  {'tpl':>4s} {'chars':>6s} {'jb':>5s} " + " ".join(f"{k[:10]:>10s}" for k in RULES))
for t in sorted(used, key=lambda t: jbt[t]):
    print(f"  {t:4d} {pre[t]:6d} {jbt[t]:5.2f} " + " ".join(f"{lab[t][k]:10d}" for k in RULES))

def within(s):
    a, n = [], []
    for u in np.unique(beh):
        m = beh == u
        if 0 < yh[m].sum() < m.sum(): a.append(fold(roc_auc_score(yh[m], s[m]))); n.append(m.sum())
    return float(np.average(a, weights=n))
rho_t = lambda s: float(abs(spearmanr([s[tpl == t].mean() for t in used], [jbt[t] for t in used])[0]))
oracle = np.array([yh[(tpl == tpl[i]) & (beh != beh[i])].mean() for i in range(len(yh))])
res = {"labels": {str(t): dict(lab[t], chars_before=pre[t], jb_rate=float(jbt[t])) for t in used}, "single": {}, "residual": {}}
print(f"\n{'score':22s} {'within-request AUC':>18s} {'rho_template':>12s} {'corr with w_out':>15s}")
for n, s in list(SC.items()) + [("template oracle", oracle)] + list(F.items()):
    c = float(np.corrcoef(s, SC["w_out"])[0, 1])
    res["single"][n] = dict(within_request=within(s), rho_template=rho_t(s), corr_w_out=c)
    print(f"{n:22s} {within(s):18.3f} {rho_t(s):12.3f} {c:+15.3f}")

def residual(s, keys):
    M = np.column_stack([np.ones(len(s))] + [F[k] for k in keys]); c, *_ = np.linalg.lstsq(M, s, rcond=None); return s - M @ c, 1 - np.var(s - M @ c) / np.var(s)
def loto_r2(keys):  # leave-one-template-out R^2 at the template level
    tm = np.array([SC["w_out"][tpl == t].mean() for t in used]); Fm = np.array([[F[k][tpl == t].mean() for k in keys] for t in used])
    pred = []
    for i in range(len(used)):
        m = np.arange(len(used)) != i; M = np.column_stack([np.ones(m.sum()), Fm[m]]); c, *_ = np.linalg.lstsq(M, tm[m], rcond=None)
        pred.append(np.r_[1, Fm[i]] @ c)
    return float(1 - np.sum((tm - np.array(pred)) ** 2) / np.sum((tm - tm.mean()) ** 2))
SETS = {"dilution": ["log_template_before", "log_template_after", "log_request_len"], "pressure": list(RULES),
        "dilution + pressure": ["log_template_before", "log_template_after", "log_request_len", *RULES]}
print(f"\nw_out after regressing out the features (random-activation level: within-request 0.68, rho_template 0.22)")
print(f"{'features':22s} {'R2 (prompts)':>12s} {'LOTO R2 (templates)':>19s} {'within-request AUC':>18s} {'rho_template':>12s}")
for n, keys in SETS.items():
    r, r2 = residual(SC["w_out"], keys); lo = loto_r2(keys)
    res["residual"][n] = dict(r2=float(r2), loto_r2=lo, within_request=within(r), rho_template=rho_t(r))
    print(f"{n:22s} {r2:12.3f} {lo:19.3f} {within(r):18.3f} {rho_t(r):12.3f}")
json.dump(res, open(HERE / "dilution_pressure.json", "w"), indent=1)
