#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/qwen3_32b/logs/%x-%j.out
# Qwen3-32B on its own GCG/PAIR/PEZ/GPTFuzz test cases: greedy responses (vLLM, stage-2 recipe), layer-32/48 readout
# at the last 9 template positions, gpt-4.1-mini judge (OpenRouter). One set per GPU; each step skips existing outputs.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_transfer.sbatch gcg,pair,pez,gptfuzz
set -uo pipefail
SETS=$1
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/qwen3_32b; S=$P/scripts
A=${ATTACKS:-$P/attacks}; OUT=${TRANSFER:-$P/transfer}; mkdir -p $OUT/logs
MODEL=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-32B', local_files_only=True))")
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /users/vvmoskvoretskii/MR-Eval/.env; set +a
export OPENAI_API_KEY="$OPENROUTER_API_KEY" OPENAI_BASE_URL=https://openrouter.ai/api/v1 JUDGE_MODEL=openai/gpt-4.1-mini
log() { echo "$(date -u +%FT%TZ) $*" >> $OUT/STATUS; }
INP=$OUT/qwen_inputs_own.jsonl
python3 $S/build_inputs_own.py $A $U/MR-Eval-hbattacks/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv $INP.tmp.$SLURM_JOB_ID > $OUT/logs/build_$SLURM_JOB_ID.log 2>&1 && mv $INP.tmp.$SLURM_JOB_ID $INP
log "build: $(tr '\n' ' ' < $OUT/logs/build_$SLURM_JOB_ID.log)"
one() {  # one SET GPU
  local st=$1; export CUDA_VISIBLE_DEVICES=$2
  [ -s $OUT/responses_$st.jsonl ] || { python3 $S/qwen_gen.py $MODEL $INP $st $OUT/responses_$st.jsonl > $OUT/logs/gen_$st.log 2>&1; log "gen $st rc=$?"; }
  [ -s $OUT/acts_pos_$st.pt ] || { python3 $S/qwen_acts_pos.py $MODEL $INP $OUT $st > $OUT/logs/acts_$st.log 2>&1; log "acts $st rc=$?"; }
  [ -s $OUT/classified_$st.jsonl ] || { python3 $S/qwen_judge.py $INP $OUT/responses_$st.jsonl $OUT 9 > $OUT/logs/judge_$st.log 2>&1; log "judge $st rc=$? $(tail -2 $OUT/logs/judge_$st.log | tr '\n' ' ')"; }
}
g=0; for st in ${SETS//,/ }; do one $st $g & g=$(( (g + 1) % 4 )); done; wait
log "done $SETS"
