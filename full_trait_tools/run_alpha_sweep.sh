#!/bin/bash
# Per-alpha (NON-escalating) sweep of the 3 real directions, so we can compute
# ASR-among-coherent = jailbroken / (n - degenerate) at each alpha independently.
# Each single-alpha run steers ALL 300 refused pairs at exactly that alpha and
# records final_jailbroken + final_degenerate per pair. Same 300 pairs every run
# (pool_seed 0). Runs inside the runai pod.
set -euo pipefail

W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"

MODEL=/mnt/dlabscratch1/moskvore/hf_cache/transformers/models--VityaVitalich--Llama3.1-8b-instruct/snapshots/247cb7c3baebabc46cff1fb97a3a5913e4c4511d
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
NAMES=w_persona__full,refusal_pos1_neg__full,dmu__full

export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a

for A in 0.1 0.2 0.3 0.4 0.5; do
  echo "=== $(date -u +%FT%TZ) | alpha=$A ==="
  "$PY" jailbreaking_tools/steer_fixed_direction.py \
    --benchmark harmbench \
    --directions inputs/steering_directions_layer16.pt \
    --names "$NAMES" \
    --model "$MODEL" \
    --classified_path "$CLS" \
    --output_dir "outputs_alpha_sweep/a$A" \
    --alpha_schedule "$A" \
    --n_refused 300 --pool_seed 0 \
    --batch_size 32 --judge_workers 8
done
echo "=== done: outputs_alpha_sweep/a<alpha>/harmbench/<arm>/results.jsonl ==="
