#!/usr/bin/env python3
"""Rebuild the exact user-turn prompts of Llama-3.1-8B-Instruct's persona-jailbreak
detection sets (gcg, pair, pap, gptfuzz, pez), as Dominic's collectors built them.

The pair lists come from the collectors' own functions, imported from full_trait_tools
(torch / transformers / tqdm are stubbed during import; no model code runs):
  gcg      collect_gcg_activations.build_pairs      behaviours CSV = harmbench_behaviors_text_test.csv
                                                    (that collector's --behaviors_path default)
  pair     collect_pair_activations.build_pairs     behaviours CSV = harmbench_behaviors_text_all.csv
  pap/gptfuzz/pez  collect_attack_activations.build_pairs   behaviours CSV = harmbench_behaviors_text_all.csv
Test cases = HarmBench results/{GCG,PAIR,PAP,GPTFuzz,PEZ}/llama3_1_8b/test_cases/test_cases.json,
copied from RCP into src/. No --test truncation (all manifests say test_mode=false).

prompt = sanitize_text(formatted_prompt) (PAIR: sanitize_text(prompt_content)), i.e. the raw user
content before the single-user-turn chat template, exactly as process_pair feeds it.

Verification: against the saved responses.jsonl of each set (local copies of
public/persona_jailbreaks/llama31_8b/activations/<set>/), every row's pair_id, behavior_id,
attack_type, jailbreak_idx, behavior_text and semantic_category must match, and counts must agree.

Writes llama_attack_inputs.jsonl next to this script.
"""
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest import mock

H = Path(__file__).resolve().parent
REPO = H.parents[2]
SRC = H / "src"
GT = REPO / "full_trait_output/qwen3_32b/llama_control/data"
TOOLS = REPO / "full_trait_tools"


def import_collectors():
    stubs = {n: mock.MagicMock(name=n) for n in
             ("torch", "torch.multiprocessing", "tqdm", "transformers")}
    saved = {n: sys.modules.get(n) for n in stubs}
    sys.modules.update(stubs)
    sys.path.insert(0, str(TOOLS))
    try:
        import collect_gcg_activations as gcg
        import collect_pair_activations as pair
        import collect_attack_activations as att
    finally:
        sys.path.remove(str(TOOLS))
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    return gcg, pair, att


def main():
    gcg, pair, att = import_collectors()
    csv_all = SRC / "harmbench_behaviors_text_all.csv"
    csv_test = SRC / "harmbench_behaviors_text_test.csv"
    tc = lambda a: SRC / a / "test_cases.json"

    built = {}
    built["gcg"] = gcg.build_pairs(gcg.load_test_cases(tc("GCG")), gcg.load_behaviors(csv_test))
    built["pair"] = pair.build_pairs(pair.load_test_cases(tc("PAIR")), pair.load_behaviors(csv_all))
    for s, a in (("pap", "PAP"), ("gptfuzz", "GPTFuzz"), ("pez", "PEZ")):
        built[s] = att.build_pairs(att.load_test_cases(tc(a)), att.load_behaviors(csv_all), a)

    sanitize = {"gcg": gcg.sanitize_text, "pair": pair.sanitize_text,
                "pap": att.sanitize_text, "gptfuzz": att.sanitize_text, "pez": att.sanitize_text}
    rows = []
    for s, pairs in built.items():
        for p in pairs:
            raw = p["prompt_content"] if s == "pair" else p["formatted_prompt"]
            rows.append(dict(set=s, pair_id=p["pair_id"], behavior_id=p["behavior_id"],
                             behavior_text=p["behavior_text"], semantic_category=p["semantic_category"],
                             attack_type=p["attack_type"], jailbreak_idx=p["jailbreak_idx"],
                             prompt=sanitize[s](raw)))

    # verify against the saved Llama rows
    keys = ("pair_id", "behavior_id", "attack_type", "jailbreak_idx", "behavior_text", "semantic_category")
    ok_all = True
    for s in built:
        gt = [json.loads(l) for l in open(GT / s / "responses.jsonl") if l.strip()]
        mine = [r for r in rows if r["set"] == s]
        man = json.load(open(GT / s / "manifest.json"))
        same = len(gt) == len(mine) == man["n_pairs"] and all(
            all(g[k] == m[k] for k in keys) for g, m in zip(gt, mine))
        ok_all &= same
        print(f"{s:8s} built={len(mine):5d} gt={len(gt):5d} manifest={man['n_pairs']:5d} "
              f"behaviours={len({r['behavior_id'] for r in mine}):4d} "
              f"empty_prompts={sum(not r['prompt'].strip() for r in mine)} "
              f"sanitize_changed={sum(r['prompt'] != (p['prompt_content'] if s == 'pair' else p['formatted_prompt']) for r, p in zip(mine, built[s]))} "
              f"match={'YES' if same else 'NO'}")
    if not ok_all:
        sys.exit("mismatch against saved Llama rows; not writing output")

    out = H / "llama_attack_inputs.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}  sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
