#!/usr/bin/env bash
# Steering-controls smoke run.
#   Pool:  HarmBench, originally-refused, n=300 (seed 0)
#   Alpha: 0.5 only  (the panel/dose-response endpoint)
#   Arms:  12 controls (4 types x 3 seeds) + the recipe-matched GSM8k off-domain arm
# Run on a GPU node from the persona-vector-jbing repo root. Judging is in-loop
# (GPT-4.1-mini) -> needs OPENAI_API_KEY in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root (script lives in full_trait_tools/)

# 0) Build the 12 control vectors (deterministic, CPU; from the trait matrix + detector).
python full_trait_tools/build_control_directions.py

# 1) Build the off-domain GSM8k arm (appends offdomain_gsm8k{,_resp}__full to the bundle).
python full_trait_tools/collect_gsm8k_direction.py --n 1000

# 2) Steer the controls at alpha=0.5 on the 300-pair refused pool.
ARMS=(
  random_full_s0__full       random_full_s1__full       random_full_s2__full
  random_persona_s0__full    random_persona_s1__full    random_persona_s2__full
  random_traitcombo_s0__full random_traitcombo_s1__full random_traitcombo_s2__full
  shuffled_persona_s0__full  shuffled_persona_s1__full  shuffled_persona_s2__full
  offdomain_gsm8k__full
)
NAMES=$(IFS=,; echo "${ARMS[*]}")

python jailbreaking_tools/steer_fixed_direction.py \
  --benchmark harmbench \
  --directions full_trait_output/steering_directions/control_directions_layer16.pt \
  --names "$NAMES" \
  --alpha_schedule 0.5 \
  --n_refused 300 --pool_seed 0 \
  --batch_size 32 --judge_workers 8 \
  --output_dir full_trait_output/steer_fixed_directions_controls

echo
echo "Done. Results under full_trait_output/steer_fixed_directions_controls/harmbench/<arm>/"
echo "Sync that dir back, then: python full_trait_tools/extract_controls_asr.py"
