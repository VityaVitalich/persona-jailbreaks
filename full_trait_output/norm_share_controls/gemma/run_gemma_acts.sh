#!/bin/bash
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/gemma_acts
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/fb9ae262347c3945692f09a612f8bb189def854f
PY=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.venv_gemma/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CHUNK_Q=${CHUNK_Q:-512} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$W"; "$PY" -c "import torch;print(torch.cuda.get_device_name(0))"
"$PY" gemma_acts.py "$MODEL" gemma_acts_inputs.jsonl gemma_acts.pt "${1:-32768}" 2>&1 | tee gemma_acts.log; echo ALL_DONE
