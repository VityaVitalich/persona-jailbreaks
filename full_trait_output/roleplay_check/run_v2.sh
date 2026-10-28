#!/bin/bash
# runai job: v2 synthetic set (persona-jailbreaks app:roleplay, three-group check): vLLM responses + L16 activations.
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/roleplay_check
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$W"
"$PY" cluster_v2.py acts --model "$MODEL" --inp synthetic_v2_prompts.jsonl --out synthetic_v2_acts.pt \
  --check check_rows.jsonl --stored_root /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output
"$PY" cluster_v2.py gen --model "$MODEL" --inp synthetic_v2_prompts.jsonl --out synthetic_v2_responses.jsonl
echo ALL_DONE
