#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/qwen3_32b/logs/%x-%j.out
# Persona-JB transfer attacks optimised against Qwen3-32B itself (HarmBench GCG / PEZ / PAIR / GPTFuzz, MR-Eval
# vendored code on branch harmbench-gcg-gptfuzz, paper hyperparameters from make_configs.py). Resumable: behaviours
# with a saved test_cases.json are skipped.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml -J pjb-gcg-00 pjb_attack.sbatch GCG ids_file
# GCG / PEZ / GPTFuzz: the ID list is split over the 4 GPUs (one target per GPU). PAIR: one process on all 4 GPUs
# (Mixtral-8x7B attacker+judge TP=2, Qwen target 1 GPU). GPTFuzz mutates through OpenRouter (openai/gpt-3.5-turbo);
# an iteration guard stops it at CAP mutation calls per node.
set -uo pipefail
M=$1; IDS=$2; CAP=${CAP:-8000}
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/qwen3_32b; HB=$U/MR-Eval-hbattacks/harmbench
A=${ATTACKS:-$P/attacks}; CFGDIR=${CFGDIR:-$P/configs}
OUT=$A/HarmBench_$M/qwen3_32b/test_cases; L=$A/logs/$M; mkdir -p "$OUT" "$L"
cd "$HB"; export PYTHONPATH="$HB/.." VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false; unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
if [ "$M" = GPTFuzz ]; then
  set -a; . /users/vvmoskvoretskii/MR-Eval/.env; set +a
  export OPENAI_API_KEY="$OPENROUTER_API_KEY" OPENAI_BASE_URL=https://openrouter.ai/api/v1 GPTFUZZ_MUTATOR_MODEL=openai/gpt-3.5-turbo
fi
echo "$(date -u +%FT%TZ) start $M $(basename $IDS) job $SLURM_JOB_ID on $(hostname)" >> $A/logs/STATUS
run() {  # run IDS_SUBSET LOG
  python3 -u generate_test_cases.py --method_name "$M" --experiment_name qwen3_32b --method_config_file "$CFGDIR/${M}_config.yaml" \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv --behavior_ids_subset "$1" --save_dir "$OUT" >> "$2" 2>&1
}
if [ "$M" = PAIR ]; then
  run "$IDS" "$L/${SLURM_JOB_ID}.log"
else
  T=$(mktemp -d); split -n l/4 -d "$IDS" "$T/part_"
  for g in 0 1 2 3; do [ -s "$T/part_0$g" ] && CUDA_VISIBLE_DEVICES=$g run "$T/part_0$g" "$L/${SLURM_JOB_ID}_g$g.log" & done
  if [ "$M" = GPTFuzz ]; then
    while [ -n "$(jobs -r)" ]; do
      n=$(cat $L/${SLURM_JOB_ID}_g*.log 2>/dev/null | grep -c "Iteration [0-9]*:")
      if [ "$n" -gt "$CAP" ]; then echo "$(date -u +%FT%TZ) GPTFuzz guard: $n mutation iterations > $CAP, stopping job $SLURM_JOB_ID" >> $A/logs/STATUS; pkill -f "method_name GPTFuz[z]"; fi
      sleep 60
    done
  fi
  wait
fi
echo "$(date -u +%FT%TZ) end $M $(basename $IDS) job $SLURM_JOB_ID done_total=$(ls $OUT/test_cases_individual_behaviors 2>/dev/null | wc -l)" >> $A/logs/STATUS
