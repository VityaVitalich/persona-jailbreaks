#!/usr/bin/env python3
"""SUMMARY.md + summary.json for the Qwen3-32B stage-2 run: STATUS lines, counts and judge jailbreak rates per set,
response finish reasons, judge parse failures and spend, readout check, detection table. usage: qwen_summary.py OUTDIR DEST_DIR"""
import json, os, sys, collections
od, dest = sys.argv[1:3]
J = lambda f: json.load(open(f"{od}/{f}")) if os.path.exists(f"{od}/{f}") else None
S = {"outdir": od, "status": open(f"{od}/STATUS").read().splitlines() if os.path.exists(f"{od}/STATUS") else [], "sets": {}}
for st in ("hb", "wjb", "pap"):
    f = f"{od}/classified_{st}.jsonl"
    if not os.path.exists(f): continue
    R = [json.loads(l) for l in open(f)]; d = {}
    for name, sub in (("all", R), ("human_jailbreak", [r for r in R if r["attack_type"] == "human_jailbreak"]), ("direct_request", [r for r in R if r["attack_type"] == "direct_request"])):
        if not sub or (st != "hb" and name != "all"): continue
        d[name] = dict(n=len(sub), jailbroken=sum(r["jailbroken"] is True for r in sub), parse_fail=sum(r["jailbroken"] is None for r in sub))
        d[name]["jb_rate"] = d[name]["jailbroken"] / max(1, d[name]["n"] - d[name]["parse_fail"])
    rf = f"{od}/responses_{st}.jsonl"
    if os.path.exists(rf): d["finish_reasons"] = dict(collections.Counter(json.loads(l).get("finish_reason") for l in open(rf)))
    S["sets"][st] = d
S["judge_spend_usd"] = (J("judge_spend.json") or {}).get("usd"); S["readout_check"] = {k: {x: v[x] for x in ("n", "min_cos", "median_cos")} for k, v in (J("readout_check.json") or {}).items()}
S["detection"] = J("detection.json")
json.dump(S, open(f"{dest}/summary.json", "w"), indent=1)
L = ["# Qwen3-32B persona-JB stage 2", "", f"Output dir: `{od}`", "", "## Steps", "", *[f"- {s}" for s in S["status"]], "", "## Sets (gpt-4.1-mini judge)", "",
     "| set | subset | n | jailbroken | rate | parse fail |", "|---|---|---|---|---|---|"]
for st, d in S["sets"].items():
    for name, v in d.items():
        if name != "finish_reasons": L.append(f"| {st} | {name} | {v['n']} | {v['jailbroken']} | {v['jb_rate']:.3f} | {v['parse_fail']} |")
L += ["", f"Finish reasons: {json.dumps({k: v.get('finish_reasons') for k, v in S['sets'].items()})}", "", f"Judge spend (cumulative): ${S['judge_spend_usd'] or 0:.2f}", "",
      "## Readout check (re-embedded one at a time vs stored)", "", *[f"- {k}: n={v['n']}, min cos {v['min_cos']:.5f}, median {v['median_cos']:.5f}" for k, v in S["readout_check"].items()], "", "## Detection (folded AUC)", ""]
D = S["detection"]
if D and "error" not in D:
    fams = D["transfer_families"]; L += [f"Splits used: {D['splits_used']} of 50; counts {json.dumps(D['n'])}", "", "| dictionary | HB test | " + " | ".join(fams) + " | transfer mean |", "|---|---|" + "---|" * (len(fams) + 1)]
    for k in ("PersonaJB", "Random Gaussian (3 draws)"):
        v = D[k]; L.append(f"| {k} | {v['hb']:.3f} | " + " | ".join(f"{v[f]:.3f}" for f in fams) + f" | {v['transfer_mean']:.3f} |" if fams else f"| {k} | {v['hb']:.3f} | — |")
else: L.append(f"not available: {D}")
open(f"{dest}/SUMMARY.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
