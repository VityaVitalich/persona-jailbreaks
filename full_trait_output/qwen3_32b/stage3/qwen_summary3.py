#!/usr/bin/env python3
"""Append the stage-3 results (GCG, PAIR, PEZ, GPTFuzz with test cases optimised against Gemma-4-31B, transferred to
Qwen) to SUMMARY.md and summary.json: STATUS3 lines, counts and judge jailbreak rates, readout check, judge spend,
detection with all five families. usage: qwen_summary3.py OUTDIR DEST_DIR"""
import json, os, sys, collections
od, dest = sys.argv[1:3]; LAB = "test cases optimised against Gemma-4-31B, transferred to Qwen"
J = lambda f: json.load(open(f"{od}/{f}")) if os.path.exists(f"{od}/{f}") else None
S = {"label": LAB, "status": open(f"{od}/STATUS3").read().splitlines() if os.path.exists(f"{od}/STATUS3") else [], "sets": {}}
for st in ("gcg", "pair", "pez", "gptfuzz"):
    f = f"{od}/classified_{st}.jsonl"
    if not os.path.exists(f): continue
    R = [json.loads(l) for l in open(f)]; n, jb, pf = len(R), sum(r["jailbroken"] is True for r in R), sum(r["jailbroken"] is None for r in R)
    rf = f"{od}/responses_{st}.jsonl"
    S["sets"][st] = dict(n=n, jailbroken=jb, parse_fail=pf, jb_rate=jb / max(1, n - pf), source=LAB,
                         finish_reasons=dict(collections.Counter(json.loads(l).get("finish_reason") for l in open(rf))) if os.path.exists(rf) else None)
S["readout_check"] = {k: {x: v[x] for x in ("n", "min_cos", "median_cos")} for k, v in (J("readout_check_transfer.json") or {}).items()}
S["judge_spend_usd_cumulative"] = (J("judge_spend.json") or {}).get("usd"); S["detection_all5"] = J("all5_detection.json")
sj = f"{dest}/summary.json"; base = json.load(open(sj)) if os.path.exists(sj) else {}; base["stage3_transfer_families"] = S; json.dump(base, open(sj, "w"), indent=1)
L = ["", f"## Stage 3: GCG, PAIR, PEZ, GPTFuzz ({LAB})", "", *[f"- {s}" for s in S["status"]], "",
     "| set | n | jailbroken | rate | parse fail | test cases |", "|---|---|---|---|---|---|"]
L += [f"| {st} | {v['n']} | {v['jailbroken']} | {v['jb_rate']:.3f} | {v['parse_fail']} | {LAB} |" for st, v in S["sets"].items()]
L += ["", f"Finish reasons: {json.dumps({k: v['finish_reasons'] for k, v in S['sets'].items()})}", f"Judge spend, cumulative stage 2 + 3: ${S['judge_spend_usd_cumulative'] or 0:.2f}", "",
      "Readout check (stage 3 sets): " + "; ".join(f"{k} n={v['n']} min cos {v['min_cos']:.5f}" for k, v in S["readout_check"].items()), "",
      f"### Detection with all five transfer families (folded AUC; GCG/PAIR/PEZ/GPTFuzz = {LAB}; PAP target-agnostic)", ""]
D = S["detection_all5"]
if D and "error" not in D:
    fams = D["transfer_families"]; L += [f"Splits used: {D['splits_used']} of 50; counts {json.dumps(D['n'])}", "", "| dictionary | HB test | " + " | ".join(fams) + " | transfer mean |", "|---|---|" + "---|" * (len(fams) + 1)]
    for k in ("PersonaJB", "Random Gaussian (3 draws)"):
        v = D[k]; L.append(f"| {k} | {v['hb']:.3f} | " + " | ".join(f"{v[f]:.3f}" for f in fams) + f" | {v['transfer_mean']:.3f} |")
else: L.append(f"not available: {D}")
open(f"{dest}/SUMMARY.md", "a").write("\n".join(L) + "\n"); print("\n".join(L))
