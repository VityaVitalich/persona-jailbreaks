#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/qwen3_32b/logs/%x-%j.out
# HarmBench PAIR against Qwen3-32B (persona-JB, paper settings: 20 streams x 3 steps, Mixtral-8x7B attacker + judge).
# The harmbench image's vLLM 0.9 cannot load Mixtral, so it is served from the serving image (vLLM 0.19, TP=2 on
# GPUs 0,1); the PAIR process (harmbench image) runs the Qwen target on GPUs 2,3 (Ray places it on one).
# Submit WITHOUT --environment (each srun step picks its own):   sbatch -J pjb-pair-00 pjb_pair.sbatch ids_file
set -uo pipefail
IDS=$1
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/qwen3_32b; HB=$U/MR-Eval-hbattacks/harmbench
A=${ATTACKS:-$P/attacks}; CFGDIR=${CFGDIR:-$P/configs}; OUT=$A/HarmBench_PAIR/qwen3_32b/test_cases; L=$A/logs/PAIR; mkdir -p "$OUT" "$L"
C=/users/vvmoskvoretskii/MR-Eval/container
echo "$(date -u +%FT%TZ) start PAIR $(basename $IDS) job $SLURM_JOB_ID on $(hostname)" >> $A/logs/STATUS
srun --overlap -n1 --gres=gpu:4 --environment=$C/serving.toml bash -c "CUDA_VISIBLE_DEVICES=0,1 vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tensor-parallel-size 2 --port 8020 --max-model-len 16384 --gpu-memory-utilization 0.90 > $L/mixtral_${SLURM_JOB_ID}.log 2>&1" &
SRV=$!
for i in $(seq 1 240); do curl -sf localhost:8020/health >/dev/null && break; kill -0 $SRV 2>/dev/null || break; sleep 10; done
if ! curl -sf localhost:8020/health >/dev/null; then
  echo "$(date -u +%FT%TZ) end PAIR $(basename $IDS) job $SLURM_JOB_ID FAILED: Mixtral server not up" >> $A/logs/STATUS; tail -30 $L/mixtral_${SLURM_JOB_ID}.log; kill $SRV; exit 1; fi
# one PAIR process per node: two Ray instances on one node collide at GCS startup
srun --overlap -n1 --gres=gpu:4 --environment=$C/harmbench.toml bash -c "
  cd $HB; export PYTHONPATH=$HB/.. VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false; unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
  CUDA_VISIBLE_DEVICES=2,3 python3 -u generate_test_cases.py --method_name PAIR --experiment_name qwen3_32b \
    --method_config_file $CFGDIR/PAIR_config.yaml --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --behavior_ids_subset $IDS --save_dir $OUT >> $L/${SLURM_JOB_ID}.log 2>&1"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "$(date -u +%FT%TZ) end PAIR $(basename $IDS) job $SLURM_JOB_ID done_total=$(ls $OUT/test_cases_individual_behaviors 2>/dev/null | wc -l)" >> $A/logs/STATUS
