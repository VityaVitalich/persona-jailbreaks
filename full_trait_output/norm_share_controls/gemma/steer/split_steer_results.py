#!/usr/bin/env python3
"""Gemma steering with the refusal / jailbreak directions, their persona component v_in and remainder v_out
(HarmBench, 300 refused prompts, pool seed 0, cumulative alpha 0.1-0.5, greedy, 300 tokens, gpt-4.1-mini judge).
ASR = judged jailbroken AND coherent text (Gemma gibberish guard looks_like_text of extract_gemma_steering_cells.py,
and not word-level degenerate); raw judge ASR also printed. References from the full 3117-prompt runs, restricted to
the same 300 prompts: random direction (random_full_s0), PersonaJB (w_persona), and the earlier full refusal / dmu
runs (batch composition differs, so not bit-identical). Paired differences: bootstrap over behaviours (2000).
Writes split_steer_results.json."""
import json, sys
from pathlib import Path
import numpy as np
O = Path.home() / "persona-vector-jbing"; sys.path.insert(0, str(O / "full_trait_tools"))
from extract_gemma_steering_cells import looks_like_text  # noqa: E402
HERE = Path(__file__).resolve().parent; NEW = HERE / "outputs_gemma_split/harmbench"; OLD = O / "full_trait_output/outputs_gemma_steer/harmbench"
tb = lambda x: x is True or str(x) == "True"
def load(p): return {str(r["pair_id"]): r for r in map(json.loads, open(p)) if r}
ARMS = ["refusal", "refusal_vin", "refusal_vout", "dmu", "dmu_vin", "dmu_vout", "persona_pattern"]
R = {a: load(NEW / a / "results.jsonl") for a in ARMS if (NEW / a / "results.jsonl").exists()}
ids = sorted(next(iter(R.values())))
assert all(sorted(r) == ids for r in R.values()), {a: len(r) for a, r in R.items()}
for a, d in (("random (ref)", "random_full_s0__full"), ("PersonaJB (ref)", "w_persona__full"), ("refusal, earlier run", "refusal_pos1_neg__full"), ("dmu, earlier run", "dmu__full")):
    o = load(OLD / d / "results.jsonl"); R[a] = {i: o[i] for i in ids}
beh = np.array([R["refusal"][i]["behavior_id"] for i in ids]); ub = np.unique(beh)
def vec(a, coherent=True, alpha=None):
    out = []
    for i in ids:
        r = R[a][i]; jb = tb(r["final_jailbroken"])
        if alpha is not None: jb = jb and float(r["min_alpha_to_jailbreak"]) <= alpha + 1e-9
        if coherent: jb = jb and not tb(r.get("final_degenerate")) and looks_like_text(r.get("final_response", ""))
        out.append(jb)
    return np.array(out, float)
rng = np.random.default_rng(0); B = [rng.choice(len(ub), len(ub)) for _ in range(2000)]
idx_of = {b: np.where(beh == b)[0] for b in ub}
def boot(d):
    m = [np.concatenate([idx_of[ub[j]] for j in s]) for s in B]
    v = np.array([d[i].mean() for i in m]); return float(d.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
res = {"n": len(ids), "arms": {}, "paired": {}}
print(f"n = {len(ids)} prompts, {len(ub)} behaviours\n{'arm':22s} {'ASR':>6s} {'raw judge':>9s}   cumulative ASR at alpha .1 .2 .3 .4 .5")
for a in R:
    c = vec(a); res["arms"][a] = dict(asr=c.mean(), raw=vec(a, False).mean(), by_alpha=[vec(a, alpha=x).mean() for x in (.1, .2, .3, .4, .5)])
    print(f"{a:22s} {100*c.mean():6.1f} {100*vec(a, False).mean():9.1f}   " + " ".join(f"{100*x:5.1f}" for x in res["arms"][a]["by_alpha"]))
print("\npaired differences in ASR (pp), 95% CI bootstrap over behaviours")
for x, y in [("refusal_vin", "refusal"), ("refusal_vout", "refusal"), ("refusal_vin", "refusal_vout"), ("dmu_vin", "dmu"), ("dmu_vout", "dmu"), ("dmu_vin", "dmu_vout"),
             ("refusal_vin", "random (ref)"), ("refusal_vout", "random (ref)"), ("dmu_vin", "random (ref)"), ("dmu_vout", "random (ref)"), ("persona_pattern", "random (ref)"), ("persona_pattern", "PersonaJB (ref)"), ("persona_pattern", "dmu_vin"), ("refusal", "refusal, earlier run"), ("dmu", "dmu, earlier run")]:
    if x in R and y in R:
        m, lo, hi = boot(vec(x) - vec(y)); res["paired"][f"{x} - {y}"] = (m, lo, hi)
        print(f"  {x:14s} - {y:22s} {100*m:+6.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
json.dump(res, open(HERE / "split_steer_results.json", "w"), indent=1)
