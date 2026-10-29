#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/logs/%x-%j.out
# Qwen3-32B control-prompt activations (alpaca, wiki, harmful) with qwen3_32b/sweep/qwen_acts_sweep.py in the harmbench
# image, as its detection sweep; one GPU per set.
#   sbatch -p debug --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_q1_ctrl_qwen.sbatch
set -uo pipefail
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb; Q=$P/q1_detection
export HF_HOME=$U/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-32B', local_files_only=True))")
mkdir -p $Q/ctrl_acts_qwen; G=(alpaca wiki harmful)
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g python3 -u $P/qwen3_32b/sweep/qwen_acts_sweep.py $M $Q/ctrl_acts_qwen ${G[$g]} $Q/ctrl_inputs.jsonl > $Q/logs/ctrl_qwen_$g.log 2>&1 & done; wait
grep -h ": [0-9]* in\|Error\|Traceback" $Q/logs/ctrl_qwen_*.log; ls $Q/ctrl_acts_qwen/*.npy | wc -l   # expect 48
