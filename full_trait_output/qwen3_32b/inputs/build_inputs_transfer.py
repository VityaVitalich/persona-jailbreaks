#!/usr/bin/env python3
"""Qwen3-32B inputs for GCG / PAIR / PEZ / GPTFuzz: the HarmBench test cases optimised against Gemma-4-31B
(HarmBench_{attack}/gemma4_31b), transferred to Qwen. Pairs built as collect_harmbench_attack_activations_gemma.py
build_pairs (sorted behaviour dirs, flattened prompt lists) and asserted to reproduce the Gemma rows' (pair_id,
behavior_id) order in {attack}_activations_gemma_hb. Writes qwen_inputs_transfer.jsonl (set = gcg/pair/pez/gptfuzz)."""
import csv, json
from pathlib import Path
H = Path(__file__).resolve().parent; O = H.parent.parent
beh = {b["BehaviorID"]: b for b in csv.DictReader(open(H / "harmbench_behaviors_text_all.csv", encoding="utf-8"))}
out = []
for a in ("GCG", "PAIR", "PEZ", "GPTFuzz"):
    merged = {}
    for bdir in sorted((H / f"tc_{a}").iterdir()):
        if not (bdir / "test_cases.json").exists(): continue
        for bid, ps in json.load(open(bdir / "test_cases.json")).items():
            flat = [q for p in ps for q in (p if isinstance(p, list) else [p]) if isinstance(q, str)]
            if flat: merged[bid] = merged.get(bid, []) + flat
    rows, pid = [], 0
    for bid, ps in merged.items():
        if bid not in beh: continue
        for p in ps:
            if p and p.strip():
                rows.append(dict(set=a.lower(), pair_id=pid, behavior_id=bid, behavior_text=beh[bid]["Behavior"], semantic_category=beh[bid].get("SemanticCategory", ""),
                                 attack_type=a.lower(), jailbreak_idx=-1, prompt=p.strip(), test_case_source="optimised against Gemma-4-31B (HarmBench gemma4_31b), transferred to Qwen")); pid += 1
    g = [json.loads(l) for l in open(O / f"{a.lower()}_activations_gemma_hb/classified_responses.jsonl")]
    assert [(int(r["pair_id"]), r["behavior_id"]) for r in g] == [(r["pair_id"], r["behavior_id"]) for r in rows], f"{a}: order differs from Gemma ({len(g)} vs {len(rows)})"
    out += rows; print(a, len(rows), "max chars", max(len(r["prompt"]) for r in rows))
with open(H / "qwen_inputs_transfer.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
