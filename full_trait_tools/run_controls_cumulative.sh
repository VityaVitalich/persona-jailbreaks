#!/bin/bash
# Controls, CUMULATIVE metric matching the main text (fig:attack): escalating
# alpha 0.1->0.5, ASR = fraction flipped at any alpha<=0.5, HarmBench refused,
# n=300 seed 0. 12 controls + the recipe-matched GSM8k off-domain arm (already
# in the bundle from the smoke). Runs inside the runai pod. Writes a new dir.
set -euo pipefail

W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"

MODEL=/mnt/dlabscratch1/moskvore/hf_cache/transformers/models--VityaVitalich--Llama3.1-8b-instruct/snapshots/247cb7c3baebabc46cff1fb97a3a5913e4c4511d
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
BUNDLE=inputs/control_directions_layer16.pt

export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a

NAMES=random_full_s0__full,random_full_s1__full,random_full_s2__full,random_persona_s0__full,random_persona_s1__full,random_persona_s2__full,random_traitcombo_s0__full,random_traitcombo_s1__full,random_traitcombo_s2__full,shuffled_persona_s0__full,shuffled_persona_s1__full,shuffled_persona_s2__full,offdomain_gsm8k__full

echo "=== $(date -u +%FT%TZ) | controls cumulative alpha 0.1-0.5 | n=300 ==="
exec "$PY" jailbreaking_tools/steer_fixed_direction.py \
    --benchmark harmbench \
    --directions "$BUNDLE" \
    --names "$NAMES" \
    --model "$MODEL" \
    --classified_path "$CLS" \
    --output_dir outputs_controls_cumulative \
    --alpha_schedule 0.1 0.2 0.3 0.4 0.5 \
    --n_refused 300 --pool_seed 0 \
    --batch_size 32 --judge_workers 8
