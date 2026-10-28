#!/bin/bash
# In-context identity probe: jailbreak templates / role-play framings with identity questions (probe/build_context_prompts.py).
# Runs inside a runai pod. Official meta-llama snapshot; +/- each direction at alpha 0.15 and 0.3, plus unsteered.
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache TOKENIZERS_PARALLELISM=false
"$PY" jailbreaking_tools/steer_probe.py --directions inputs/steer_probe_layer16.pt --model "$MODEL" \
    --names w_out,w_in,persona --prompts_file inputs/context_prompts.json --alphas 0.15 0.3 --out outputs_probe/probe_context.jsonl
