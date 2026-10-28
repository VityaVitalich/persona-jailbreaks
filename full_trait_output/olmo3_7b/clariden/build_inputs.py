#!/usr/bin/env python3
"""OLMo-3-7B-Instruct inputs. Shared sets hb / wjb / pap: the qwen_inputs.jsonl rows unchanged (same prompts and pair ids
as the Gemma / Qwen sets). Own sets gcg / pair / pez / gptfuzz: the test cases optimised against OLMo-3-7B-Instruct
(attacks/HarmBench_{A}/olmo3_7b), flattened as build_inputs_own.py does for Qwen (sorted behaviour dirs, flattened
prompt lists); sets with no test cases are skipped.   usage: build_inputs.py SHARED_JSONL ATTACKS_DIR BEHAVIORS_CSV OUT"""
import csv, json, sys
from pathlib import Path
SHARED, A_DIR, CSV, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
beh = {b["BehaviorID"]: b for b in csv.DictReader(open(CSV, encoding="utf-8"))}
out = [r for r in map(json.loads, open(SHARED)) if r["set"] in ("hb", "wjb", "pap")]
print("shared", {s: sum(r["set"] == s for r in out) for s in ("hb", "wjb", "pap")})
for a in ("GCG", "PAIR", "PEZ", "GPTFuzz"):
    d = A_DIR / f"HarmBench_{a}/olmo3_7b/test_cases/test_cases_individual_behaviors"
    if not d.exists(): print(a, "missing"); continue
    merged = {}
    for bdir in sorted(d.iterdir()):
        if not (bdir / "test_cases.json").exists(): continue
        for bid, ps in json.load(open(bdir / "test_cases.json")).items():
            flat = [q for p in ps for q in (p if isinstance(p, list) else [p]) if isinstance(q, str)]
            if flat: merged[bid] = merged.get(bid, []) + flat
    n0 = len(out); k = 0
    for bid, ps in merged.items():
        if bid not in beh: continue
        for p in ps:
            if p and p.strip():
                out.append(dict(set=a.lower(), pair_id=k, behavior_id=bid, behavior_text=beh[bid]["Behavior"],
                                semantic_category=beh[bid].get("SemanticCategory", ""), attack_type=a.lower(), jailbreak_idx=-1,
                                prompt=p.strip(), test_case_source="optimised against OLMo-3-7B-Instruct (HarmBench olmo3_7b)"))
                k += 1
    print(a, "behaviours", len(merged), "rows", len(out) - n0)
with open(OUT, "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
