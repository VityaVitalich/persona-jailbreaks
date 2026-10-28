#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/olmo3_7b/logs/%x-%j.out
# Persona-JB PAIR / GPTFuzz against OLMo-3-7B-Instruct. The harmbench image's vLLM 0.9 cannot load OLMo-3 (nor Mixtral),
# so the target is served from the serving image (vLLM 0.19, port 8030) and HarmBench (harmbench image) queries it with
# prompts rendered by OLMo's chat template (target base_url, make_configs.py).
#   PAIR:    Mixtral-8x7B attacker+judge TP=2 on GPUs 0,1 (port 8020), OLMo on GPU 2, one PAIR process (two Ray
#            instances on one node collide at GCS startup). Paper settings 20 streams x 3 steps.
#   GPTFuzz: OLMo on GPU 0; the ID list split over 4 processes whose RoBERTa judges sit on GPUs 1,2,3,1. Mutator
#            openai/gpt-3.5-turbo through OpenRouter; a guard stops the job above CAP mutation iterations.
# Submit WITHOUT --environment (each srun step picks its own):   sbatch -J pjb-pair-00 pjb_served.sbatch PAIR ids_file
set -uo pipefail
M=$1; IDS=$2; CAP=${CAP:-8000}
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/olmo3_7b; HB=$U/MR-Eval-hbattacks/harmbench
A=${ATTACKS:-$P/attacks}; CFGDIR=${CFGDIR:-$P/configs}; OUT=$A/HarmBench_$M/olmo3_7b/test_cases; L=$A/logs/$M; mkdir -p "$OUT" "$L"
C=/users/vvmoskvoretskii/MR-Eval/container
# HF_HOME is exported inside every srun step: an export here is undone when the step's bash starts (the a141 cache from
# the user's shell init wins over both this shell and the TOML), and OLMo is cached in infra01
H="export HF_HOME=$U/hf_cache"
echo "$(date -u +%FT%TZ) start $M $(basename $IDS) job $SLURM_JOB_ID on $(hostname)" >> $A/logs/STATUS
fail() { echo "$(date -u +%FT%TZ) end $M $(basename $IDS) job $SLURM_JOB_ID FAILED: $1" >> $A/logs/STATUS; kill $(jobs -p) 2>/dev/null; exit 1; }
up() { for i in $(seq 1 240); do curl -sf localhost:$1/health >/dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 10; done; return 1; }
OG=$([ "$M" = PAIR ] && echo 2 || echo 0)
srun --overlap -n1 --gres=gpu:4 --environment=$C/serving.toml bash -c "$H; CUDA_VISIBLE_DEVICES=$OG vllm serve allenai/Olmo-3-7B-Instruct \
  --port 8030 --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching > $L/olmo_${SLURM_JOB_ID}.log 2>&1" &
OLMO=$!
if [ "$M" = PAIR ]; then
  srun --overlap -n1 --gres=gpu:4 --environment=$C/serving.toml bash -c "$H; CUDA_VISIBLE_DEVICES=0,1 vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 2 --port 8020 --max-model-len 16384 --gpu-memory-utilization 0.90 > $L/mixtral_${SLURM_JOB_ID}.log 2>&1" &
  MIX=$!; up 8020 $MIX || { tail -30 $L/mixtral_${SLURM_JOB_ID}.log; fail "Mixtral server not up"; }
fi
up 8030 $OLMO || { tail -30 $L/olmo_${SLURM_JOB_ID}.log; fail "OLMo server not up"; }
SETUP="$H; cd $HB; export PYTHONPATH=$HB/.. VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false; unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE"
gen() {  # gen IDS_SUBSET GPU LOG
  echo "$SETUP; CUDA_VISIBLE_DEVICES=$2 python3 -u generate_test_cases.py --method_name $M --experiment_name olmo3_7b \
    --method_config_file $CFGDIR/${M}_config.yaml --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --behavior_ids_subset $1 --save_dir $OUT >> $3 2>&1"
}
if [ "$M" = PAIR ]; then
  srun --overlap -n1 --gres=gpu:4 --environment=$C/harmbench.toml bash -c "$(gen $IDS 3 $L/${SLURM_JOB_ID}.log)"
else
  T=$A/logs/parts_$SLURM_JOB_ID; mkdir -p $T; split -n l/4 -d "$IDS" "$T/part_"
  CMD="set -a; . /users/vvmoskvoretskii/MR-Eval/.env; set +a
    export OPENAI_API_KEY=\"\$OPENROUTER_API_KEY\" OPENAI_BASE_URL=https://openrouter.ai/api/v1 GPTFUZZ_MUTATOR_MODEL=openai/gpt-3.5-turbo"
  G=(1 2 3 1); for g in 0 1 2 3; do [ -s "$T/part_0$g" ] && CMD="$CMD
    ( $(gen $T/part_0$g ${G[$g]} $L/${SLURM_JOB_ID}_g$g.log) ) &"; done
  srun --overlap -n1 --gres=gpu:4 --environment=$C/harmbench.toml bash -c "$CMD
    wait" &
  RUN=$!
  while kill -0 $RUN 2>/dev/null; do
    n=$(cat $L/${SLURM_JOB_ID}_g*.log 2>/dev/null | grep -c "Iteration [0-9]*:")
    if [ "$n" -gt "$CAP" ]; then echo "$(date -u +%FT%TZ) GPTFuzz guard: $n mutation iterations > $CAP, stopping job $SLURM_JOB_ID" >> $A/logs/STATUS; kill $RUN; break; fi
    sleep 60
  done
  wait $RUN 2>/dev/null
fi
kill $(jobs -p) 2>/dev/null; wait 2>/dev/null
echo "$(date -u +%FT%TZ) end $M $(basename $IDS) job $SLURM_JOB_ID done_total=$(ls $OUT/test_cases_individual_behaviors 2>/dev/null | wc -l) plain_template_fallbacks=$(cat $L/${SLURM_JOB_ID}*.log 2>/dev/null | grep -c "Plain instruction template")" >> $A/logs/STATUS
