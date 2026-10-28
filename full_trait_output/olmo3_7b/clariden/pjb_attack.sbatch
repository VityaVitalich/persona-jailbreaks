#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/olmo3_7b/logs/%x-%j.out
# Persona-JB GCG / PEZ optimised against OLMo-3-7B-Instruct itself (MR-Eval vendored HarmBench, branch
# harmbench-gcg-gptfuzz, paper hyperparameters from make_configs.py). The ID list is split over the 4 GPUs, one target
# per GPU. Resumable: behaviours with a saved test_cases.json are skipped.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml -J pjb-gcg-00 pjb_attack.sbatch GCG ids_file
set -uo pipefail
M=$1; IDS=$2
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/olmo3_7b; HB=$U/MR-Eval-hbattacks/harmbench
A=${ATTACKS:-$P/attacks}; CFGDIR=${CFGDIR:-$P/configs}
OUT=$A/HarmBench_$M/olmo3_7b/test_cases; L=$A/logs/$M; mkdir -p "$OUT" "$L"
cd "$HB"; export PYTHONPATH="$HB/.." VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false; unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export HF_HOME=$U/hf_cache  # the submitting shell's HF_HOME (a141) would override the TOML; OLMo is cached here
echo "$(date -u +%FT%TZ) start $M $(basename $IDS) job $SLURM_JOB_ID on $(hostname)" >> $A/logs/STATUS
T=$(mktemp -d); split -n l/4 -d "$IDS" "$T/part_"
for g in 0 1 2 3; do
  [ -s "$T/part_0$g" ] && CUDA_VISIBLE_DEVICES=$g python3 -u generate_test_cases.py --method_name "$M" --experiment_name olmo3_7b \
    --method_config_file "$CFGDIR/${M}_config.yaml" --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --behavior_ids_subset "$T/part_0$g" --save_dir "$OUT" >> "$L/${SLURM_JOB_ID}_g$g.log" 2>&1 &
done
wait
echo "$(date -u +%FT%TZ) end $M $(basename $IDS) job $SLURM_JOB_ID done_total=$(ls $OUT/test_cases_individual_behaviors 2>/dev/null | wc -l) plain_template_fallbacks=$(cat $L/${SLURM_JOB_ID}_g*.log 2>/dev/null | grep -c "Plain instruction template")" >> $A/logs/STATUS
