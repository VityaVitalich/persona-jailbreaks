#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/qwen3_32b/sweep/logs/%x-%j.out
# Layer x position sweep: (1) readout of all 7 sets at 16 layers x 10 positions, one GPU per set group; (2) detector sweep on CPU.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_sweep.sbatch
set -uo pipefail
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/qwen3_32b; D=$P/sweep; TR=$P/transfer
MODEL=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-32B', local_files_only=True))")
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=(hb wjb,gcg,pair pap,gptfuzz pez)
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python3 -u $D/qwen_acts_sweep.py $MODEL $D/acts ${G[$g]} $D/in/qwen_inputs.jsonl $TR/qwen_inputs_own.jsonl > $D/logs/acts_$g.log 2>&1 & done; wait
n=$(ls $D/acts/*.npy 2>/dev/null | wc -l); echo "$(date -u +%FT%TZ) acts done: $n files" >> $D/STATUS
[ "$n" -eq 112 ] || { echo "$(date -u +%FT%TZ) acts incomplete, stop" >> $D/STATUS; exit 1; }
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 python3 -u $D/detect_sweep.py $D $TR 16 > $D/logs/detect.log 2>&1
echo "$(date -u +%FT%TZ) detect rc=$?" >> $D/STATUS
