#!/bin/bash
# Steering-controls smoke, run INSIDE the runai pod (mirrors run_arm.sh).
#   Pool:  HarmBench originally-refused, n=300 (seed 0)   Alpha: 0.5 only
#   Arms:  12 controls (4 types x 3 seeds) + recipe-matched GSM8k off-domain arm
# Writes to outputs_controls/ so the canonical outputs/ tree is untouched.
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
set -a; . "$W/openai.env"; set +a          # judge key, never echoed

"$PY" -c "import torch,transformers,openai,datasets,sklearn; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== $(date -u +%FT%TZ) | step 1: GSM8k off-domain direction ==="
"$PY" full_trait_tools/collect_gsm8k_direction.py --n 1000 --model "$MODEL" --bundle "$BUNDLE"

NAMES=random_full_s0__full,random_full_s1__full,random_full_s2__full,random_persona_s0__full,random_persona_s1__full,random_persona_s2__full,random_traitcombo_s0__full,random_traitcombo_s1__full,random_traitcombo_s2__full,shuffled_persona_s0__full,shuffled_persona_s1__full,shuffled_persona_s2__full,offdomain_gsm8k__full

echo "=== $(date -u +%FT%TZ) | step 2: steer controls | alpha=0.5 n=300 ==="
exec "$PY" jailbreaking_tools/steer_fixed_direction.py \
    --benchmark harmbench \
    --directions "$BUNDLE" \
    --names "$NAMES" \
    --model "$MODEL" \
    --classified_path "$CLS" \
    --output_dir outputs_controls \
    --alpha_schedule 0.5 \
    --n_refused 300 --pool_seed 0 \
    --batch_size 32 --judge_workers 8
