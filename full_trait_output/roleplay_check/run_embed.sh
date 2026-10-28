#!/bin/bash
# runai job: L16 activations for the synthetic role-play verification set (persona-jailbreaks).
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/roleplay_check
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$W"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exec "$PY" embed_cluster.py --model "$MODEL" --inp synthetic_prompts.jsonl --check check_rows.jsonl \
  --stored_root /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output --out synthetic_acts.pt
