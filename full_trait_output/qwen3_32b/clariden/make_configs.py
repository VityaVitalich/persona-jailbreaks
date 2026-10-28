#!/usr/bin/env python3
"""Method configs for the persona-JB Qwen3-32B attacks: the vendored MR-Eval HarmBench configs with the HarmBench
defaults the paper's Llama/Gemma test cases used (PEZ 3 test cases x 100 steps; PAIR 20 streams x 3 steps; GCG and
GPTFuzz unchanged), plus a qwen3_32b target block. --smoke: same code paths, a few steps.
usage: make_configs.py HARMBENCH_DIR OUT_DIR [--smoke]"""
import sys, yaml
from pathlib import Path
HB, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
TGT = {"model_name_or_path": "Qwen/Qwen3-32B", "use_fast_tokenizer": True, "dtype": "bfloat16", "num_gpus": 1}
PAPER = {"PEZ": {"num_test_cases_per_behavior": 3, "test_cases_batch_size": 3, "num_steps": 100},
         "PAIR": {"n_streams": 20, "steps": 3}, "GCG": {}, "GPTFuzz": {}}
if "--smoke" in sys.argv:
    for m, o in {"PEZ": {"num_steps": 4}, "PAIR": {"n_streams": 2, "steps": 1}, "GCG": {"num_steps": 4, "eval_steps": 2}, "GPTFuzz": {"max_query": 6, "max_iteration": 2}}.items(): PAPER[m] = {**PAPER[m], **o}
BLOCK = {"PEZ": {"target_model": TGT}, "GCG": {"target_model": TGT},
         "GPTFuzz": {"target_model": {k: v for k, v in TGT.items() if k != "use_fast_tokenizer"}, "num_gpus": 1},
         "PAIR": {"target_model": {"model_name_or_path": "Qwen/Qwen3-32B", "use_vllm": True, "num_gpus": 1, "dtype": "bfloat16",
                                   "use_tokenizer_chat_template": True},
                  # Mixtral attacker + judge served from the serving container (pjb_pair.sbatch); same raw prompts
                  "attack_model": {"model_name_or_path": "mistralai/Mixtral-8x7B-Instruct-v0.1", "base_url": "http://localhost:8020/v1"},
                  "judge_model": {"model_name_or_path": "mistralai/Mixtral-8x7B-Instruct-v0.1", "base_url": "http://localhost:8020/v1"}}}
for m in PAPER:
    c = yaml.full_load(open(HB / f"configs/method_configs/{m}_config.yaml"))
    c["default_method_hyperparameters"].update(PAPER[m])
    c["qwen3_32b"] = BLOCK[m]
    yaml.safe_dump(c, open(OUT / f"{m}_config.yaml", "w"), sort_keys=False)
    d = c["default_method_hyperparameters"]
    print(m, {k: d[k] for k in ("num_steps", "search_width", "num_test_cases_per_behavior", "n_streams", "steps", "max_iteration", "mutate_model_path") if k in d})
