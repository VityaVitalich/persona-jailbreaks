#!/bin/bash
# runai job: Llama-3.1-8B responses for the synthetic role-play set (persona-jailbreaks, app:roleplay).
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/roleplay_check
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$W"
exec "$PY" generate_cluster.py --model "$MODEL" --inp synthetic_prompts.jsonl --check gen_check_rows.jsonl --out synthetic_responses.jsonl
