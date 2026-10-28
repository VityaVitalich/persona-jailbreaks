#!/usr/bin/env python3
"""Rebuild the exact user-turn prompts of Gemma-4-31B-it's persona-jailbreak detection sets
(gcg, pair, gptfuzz, pez, pap), as Dominic's collector built them.

All five sets come from one collector, full_trait_tools/collect_harmbench_attack_activations_gemma.py
(identical, md5 b3200404, to the RCP copy in bazina/assistant-axis-llama3.1-8B). Its own
load_individual_test_cases / load_behaviors / build_pairs / sanitize_text are imported with torch,
transformers, tqdm and chunked_sdpa stubbed (no model code runs), and HARMBENCH_ROOT is pointed at
src/HarmBench, a read-only copy of the RCP layout:

  set      --attack_type  --exp (run_collect_*_gemma.sh / run_gcg_gemma_finish.sh, = manifest experiment_name)
  gcg      GCG            gemma4_31b      (gcg_gemma.py shards -> gcg_gemma_to_harmbench.py, 159 behaviours)
  pair     PAIR           gemma4_31b
  gptfuzz  GPTFuzz        gemma4_31b
  pez      PEZ            gemma4_31b
  pap      PAP            olmo3_7b_top5   (target-agnostic PAP test cases)

  test cases  src/HarmBench/results/HarmBench_<attack>/<exp>/test_cases/test_cases_individual_behaviors/*/test_cases.json
  behaviours  src/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv  (collector --behaviors_path default)
No --test truncation (all manifests say test_mode=false). Every test_cases.json predates its set's manifest.

prompt = sanitize_text(formatted_prompt), where formatted_prompt = test_case.strip() (build_pairs): the user
content exactly as get_prompt_input_ids passes it to the single-user-turn chat template.

Verification, per set: against the saved classified_responses.jsonl (local copy in ../gemma_data/<set>/) and
pairs_metadata.jsonl (copy of public/persona_jailbreaks/gemma4_31b/activations/<set>/ in src/gt/<set>/),
row by row, pair_id (value and JSON type), behavior_id, attack_type, jailbreak_idx, behavior_text and
semantic_category must match; counts must equal manifest n_pairs, and n_behaviors / experiment_name must
match the manifest. Any mismatch -> nothing is written.

PAP is rebuilt and verified but NOT written: the set=="pap" rows of ../../qwen3_32b/inputs/qwen_inputs.jsonl
(qwen3_32b/inputs/build_inputs.py, same olmo3_7b_top5 test cases and CSV) already hold the identical rows
(all fields and prompt strings); this script asserts that and fails otherwise.

Writes gemma_attack_inputs.jsonl (gcg, pair, gptfuzz, pez) next to this script.
"""
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

H = Path(__file__).resolve().parent
REPO = H.parents[2]
SRC = H / "src"
GT = H.parent / "gemma_data"
TOOLS = REPO / "full_trait_tools"
QWEN_INPUTS = REPO / "full_trait_output/qwen3_32b/inputs/qwen_inputs.jsonl"

SETS = (("gcg", "GCG", "gemma4_31b"), ("pair", "PAIR", "gemma4_31b"), ("gptfuzz", "GPTFuzz", "gemma4_31b"),
        ("pez", "PEZ", "gemma4_31b"), ("pap", "PAP", "olmo3_7b_top5"))
KEYS = ("pair_id", "behavior_id", "attack_type", "jailbreak_idx", "behavior_text", "semantic_category")


def import_collector():
    stubs = {n: mock.MagicMock(name=n) for n in
             ("torch", "tqdm", "transformers", "chunked_sdpa")}
    saved = {n: sys.modules.get(n) for n in stubs}
    sys.modules.update(stubs)
    sys.path.insert(0, str(TOOLS))
    try:
        import collect_harmbench_attack_activations_gemma as col
    finally:
        sys.path.remove(str(TOOLS))
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    col.HARMBENCH_ROOT = SRC / "HarmBench"
    return col


def same(a, b):
    return all(a[k] == b[k] and type(a[k]) is type(b[k]) for k in KEYS)


def main():
    col = import_collector()
    behaviors = col.load_behaviors(SRC / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")

    rows, ok_all = [], True
    for s, attack, exp in SETS:
        man = json.load(open(GT / s / "manifest.json"))
        assert man["attack_type"] == attack and man["test_mode"] is False, s
        tc = col.load_individual_test_cases(attack, exp)
        pairs = col.build_pairs(tc, behaviors, attack)
        mine = [dict(set=s, pair_id=p["pair_id"], behavior_id=p["behavior_id"], behavior_text=p["behavior_text"],
                     semantic_category=p["semantic_category"], attack_type=p["attack_type"],
                     jailbreak_idx=p["jailbreak_idx"], prompt=col.sanitize_text(p["formatted_prompt"]))
                for p in pairs]

        gt_cls = [json.loads(l) for l in open(GT / s / "classified_responses.jsonl") if l.strip()]
        gt_meta = [json.loads(l) for l in open(SRC / "gt" / s / "pairs_metadata.jsonl") if l.strip()]
        ok = (exp == man["experiment_name"] and len(tc) == man["n_behaviors"]
              and len(mine) == len(gt_cls) == len(gt_meta) == man["n_pairs"]
              and all(same(g, m) for g, m in zip(gt_cls, mine))
              and all(same(g, m) for g, m in zip(gt_meta, mine)))
        ok_all &= ok
        print(f"{s:8s} exp={exp:14s} built={len(mine):5d} gt={len(gt_cls):5d} meta={len(gt_meta):5d} "
              f"manifest={man['n_pairs']:5d} behaviours={len(tc):4d}/{man['n_behaviors']:4d} "
              f"empty={sum(not r['prompt'].strip() for r in mine)} "
              f"sanitize_changed={sum(r['prompt'] != p['formatted_prompt'] for r, p in zip(mine, pairs))} "
              f"match={'YES' if ok else 'NO'}")
        rows += mine

    pap = [r for r in rows if r["set"] == "pap"]
    q = [json.loads(l) for l in open(QWEN_INPUTS) if l.strip()]
    q = [r for r in q if r["set"] == "pap"]
    pap_same = len(q) == len(pap) and all(same(a, b) and a["prompt"] == b["prompt"] for a, b in zip(q, pap))
    print(f"pap      identical to qwen_inputs.jsonl set==pap rows (ids + prompts): {'YES' if pap_same else 'NO'}")
    if not (ok_all and pap_same):
        sys.exit("mismatch against saved Gemma rows; not writing output")
    rows = [r for r in rows if r["set"] != "pap"]

    out = H / "gemma_attack_inputs.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}  sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
