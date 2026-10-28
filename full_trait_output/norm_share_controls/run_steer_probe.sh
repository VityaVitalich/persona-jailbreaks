#!/bin/bash
# Behavioural probe of steering directions (norm_share_controls/build_steer_probe.py bundle, jailbreaking_tools/steer_probe.py).
# Runs inside a runai pod. Official meta-llama snapshot; +/- each direction at alpha 0.15 and 0.3, plus unsteered.
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache TOKENIZERS_PARALLELISM=false
"$PY" jailbreaking_tools/steer_probe.py --directions inputs/steer_probe_layer16.pt --model "$MODEL" \
    --alphas 0.15 0.3 --out outputs_probe/probe.jsonl
