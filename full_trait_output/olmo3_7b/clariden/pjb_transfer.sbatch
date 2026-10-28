#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/olmo3_7b/logs/%x-%j.out
# OLMo-3-7B-Instruct on the shared sets (hb/wjb/pap) and its own GCG/PAIR/PEZ/GPTFuzz test cases: greedy responses (vLLM),
# readout at layers 4..31 x the 5 template-tail positions, gpt-4.1-mini judge (OpenRouter). One set at a time per GPU;
# each step skips existing outputs. Serving image (vLLM 0.19, transformers 4.57: native OLMo-3).
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/serving.toml pjb_transfer.sbatch hb,wjb,pap,gcg,pair,pez,gptfuzz
set -uo pipefail
SETS=$1
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/olmo3_7b; S=$P/scripts
export HF_HOME=$U/hf_cache  # the submitting shell's HF_HOME (a141) would override the TOML; OLMo is cached here
A=${ATTACKS:-$P/attacks}; OUT=${TRANSFER:-$P/transfer}; mkdir -p $OUT/logs
MODEL=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('allenai/Olmo-3-7B-Instruct', local_files_only=True))")
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /users/vvmoskvoretskii/MR-Eval/.env; set +a
export OPENAI_API_KEY="$OPENROUTER_API_KEY" OPENAI_BASE_URL=https://openrouter.ai/api/v1 JUDGE_MODEL=openai/gpt-4.1-mini
log() { echo "$(date -u +%FT%TZ) $*" >> $OUT/STATUS; }
INP=$OUT/olmo_inputs.jsonl
python3 $S/build_inputs.py $S/shared_inputs.jsonl $A $U/MR-Eval-hbattacks/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv $INP.tmp.$SLURM_JOB_ID > $OUT/logs/build_$SLURM_JOB_ID.log 2>&1 && mv $INP.tmp.$SLURM_JOB_ID $INP
log "build: $(tr '\n' ' ' < $OUT/logs/build_$SLURM_JOB_ID.log)"
one() {  # one SET GPU
  local st=$1; export CUDA_VISIBLE_DEVICES=$2
  [ -s $OUT/responses_$st.jsonl ] || { python3 $S/gen.py $MODEL $INP $st $OUT/responses_$st.jsonl ${TEST:-} > $OUT/logs/gen_$st.log 2>&1; log "gen $st rc=$?"; }
  [ -s $OUT/acts_$st.pt ] || { python3 $S/acts.py $MODEL $INP $OUT $st ${TEST:-} > $OUT/logs/acts_$st.log 2>&1; log "acts $st rc=$?"; }
  [ -s $OUT/classified_$st.jsonl ] || { python3 $S/judge.py $INP $OUT/responses_$st.jsonl $OUT 9 > $OUT/logs/judge_$st.log 2>&1; log "judge $st rc=$? $(tail -2 $OUT/logs/judge_$st.log | tr '\n' ' ')"; }
}
# round-robin the sets over the GPUs; sets on the same GPU run one after another
for g in 0 1 2 3; do ( i=0; for st in ${SETS//,/ }; do [ $((i % 4)) -eq $g ] && one $st $g; i=$((i + 1)); done ) & done; wait
log "done $SETS"
