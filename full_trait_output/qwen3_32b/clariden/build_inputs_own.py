#!/usr/bin/env python3
"""Qwen3-32B inputs for GCG / PAIR / PEZ / GPTFuzz from the test cases optimised against Qwen3-32B itself
(attacks/HarmBench_{A}/qwen3_32b). Same flattening and pair_id order as build_inputs_transfer.py (sorted behaviour
dirs, flattened prompt lists). Writes qwen_inputs_own.jsonl (set = gcg/pair/pez/gptfuzz); sets with no test cases
are skipped.   usage: build_inputs_own.py ATTACKS_DIR BEHAVIORS_CSV OUT"""
import csv, json, sys
from pathlib import Path
A_DIR, CSV, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
beh = {b["BehaviorID"]: b for b in csv.DictReader(open(CSV, encoding="utf-8"))}
out = []
for a in ("GCG", "PAIR", "PEZ", "GPTFuzz"):
    d = A_DIR / f"HarmBench_{a}/qwen3_32b/test_cases/test_cases_individual_behaviors"
    if not d.exists(): print(a, "missing"); continue
    merged = {}
    for bdir in sorted(d.iterdir()):
        if not (bdir / "test_cases.json").exists(): continue
        for bid, ps in json.load(open(bdir / "test_cases.json")).items():
            flat = [q for p in ps for q in (p if isinstance(p, list) else [p]) if isinstance(q, str)]
            if flat: merged[bid] = merged.get(bid, []) + flat
    n0 = len(out)
    for bid, ps in merged.items():
        if bid not in beh: continue
        for p in ps:
            if p and p.strip():
                out.append(dict(set=a.lower(), pair_id=len(out) - n0, behavior_id=bid, behavior_text=beh[bid]["Behavior"],
                                semantic_category=beh[bid].get("SemanticCategory", ""), attack_type=a.lower(), jailbreak_idx=-1,
                                prompt=p.strip(), test_case_source="optimised against Qwen3-32B (HarmBench qwen3_32b)"))
    print(a, "behaviours", len(merged), "rows", len(out) - n0)
with open(OUT, "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
