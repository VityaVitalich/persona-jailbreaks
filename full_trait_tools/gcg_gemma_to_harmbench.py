#!/usr/bin/env python3
"""
gcg_gemma_to_harmbench.py

Convert gcg_gemma.py shard output into the per-behaviour test-case layout that
collect_harmbench_attack_activations_gemma.py reads:

  HarmBench/results/HarmBench_GCG/<exp>/test_cases/test_cases_individual_behaviors/
      <behavior_id>/test_cases.json     ->  {"<behavior_id>": ["<test case>"]}
"""
import argparse
import json
from pathlib import Path

HARMBENCH_ROOT = Path("/mnt/dlabscratch1/bazina/HarmBench")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", default="full_trait_output/gcg_gemma_test_cases")
    p.add_argument("--exp", default="gemma4_31b")
    args = p.parse_args()

    out_root = (HARMBENCH_ROOT / "results" / "HarmBench_GCG" / args.exp
                / "test_cases" / "test_cases_individual_behaviors")
    out_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for shard in sorted(Path(args.shard_dir).glob("shard*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            bid = rec["behavior_id"]
            bdir = out_root / bid
            bdir.mkdir(exist_ok=True)
            (bdir / "test_cases.json").write_text(json.dumps({bid: [rec["test_case"]]}, indent=2))
            n += 1

    print(f"Wrote {n} behaviours to {out_root}")


if __name__ == "__main__":
    main()
