#!/bin/bash
# One job of the Gemma-4-31B steering reproduction (B300 pool).
#   bash run_gemma_steer.sh <benchmark> <comma-separated-arms> [extra runner args...]
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
BENCH="${1:?usage: run_gemma_steer.sh <benchmark> <arms> [extra]}"; NAMES="${2:?}"; shift 2
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/fb9ae262347c3945692f09a612f8bb189def854f
PY=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.venv_gemma/bin/python
BUNDLE=inputs/gemma/steering_directions_gemma_layer30.pt
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CHUNK_Q=${CHUNK_Q:-512}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . "$W/openai.env"; set +a          # judge key, never echoed
mkdir -p logs/gemma_steer
LOG=logs/gemma_steer/${BENCH}_$(echo "$NAMES" | tr ',' '+' | cut -c1-80)_$(date -u +%Y%m%dT%H%M%SZ).log
echo "=== $(date -u +%FT%TZ) | gemma steer | $BENCH | arms: $NAMES | extra: $* ===" | tee "$LOG"
"$PY" -c "import torch;print('torch',torch.__version__,'| gpu',torch.cuda.get_device_name(0),'| cap',torch.cuda.get_device_capability(0),'| archs',torch.cuda.get_arch_list())" 2>&1 | tee -a "$LOG"
exec "$PY" jailbreaking_tools/steer_fixed_direction_gemma.py \
    --benchmark "$BENCH" --directions "$BUNDLE" --names "$NAMES" --model "$MODEL" \
    --output_dir "${OUT_DIR:-outputs_gemma_steer}" "$@" 2>&1 | tee -a "$LOG"
