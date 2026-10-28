#!/bin/bash
# Sampled behavioural probes (8 samples per prompt, temperature 0.7) of the steering directions in
# inputs/steer_probe_layer16.pt (norm_share_controls/build_steer_probe.py). Runs inside a runai pod.
# Usage: bash run_steer_probe_sampled.sh <shard plain|ctx_out|ctx_rest>
#   plain     the 30 built-in probe questions, all 6 directions + unsteered
#   ctx_out   identity questions inside jailbreak templates / role-play framings (inputs/context_prompts.json), w_out + unsteered
#   ctx_rest  the same prompts, w_in and persona (no unsteered baseline)
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache TOKENIZERS_PARALLELISM=false
COMMON=(--directions inputs/steer_probe_layer16.pt --model "$MODEL" --alphas 0.15 0.3 --samples 8 --temperature 0.7)
case "$1" in
  plain) "$PY" jailbreaking_tools/steer_probe.py "${COMMON[@]}" --chunk 16 --out outputs_probe/probe_sampled.jsonl ;;
  ctx_out) "$PY" jailbreaking_tools/steer_probe.py "${COMMON[@]}" --names w_out --prompts_file inputs/context_prompts.json --chunk 4 --out outputs_probe/probe_context_sampled_wout.jsonl ;;
  ctx_rest) "$PY" jailbreaking_tools/steer_probe.py "${COMMON[@]}" --names w_in,persona --skip_none --seed 1 --prompts_file inputs/context_prompts.json --chunk 4 --out outputs_probe/probe_context_sampled_rest.jsonl ;;
  *) echo "shard?"; exit 1 ;;
esac
