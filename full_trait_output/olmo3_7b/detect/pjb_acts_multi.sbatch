#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/hb_select/logs/%x-%j.out
# Multi-layer/position detection activations (acts_multi.py), 4 shards on 4 GPUs, then merge. Serving image.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/serving.toml pjb_acts_multi.sbatch MODEL INPUTS OUTDIR LAYERS [--gemma]
# OVERLAY (optional): a directory prepended to PYTHONPATH (Gemma: transformers 5.8 overlay).
set -uo pipefail
MODEL=$1; INP=$2; OUT=$3; LAYERS=$4; EXTRA=${5:-}
H=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/hb_select
export HF_HOME=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "${OVERLAY:-}" ] && export PYTHONPATH=$OVERLAY${PYTHONPATH:+:$PYTHONPATH}
SETS=${SETS:-hb,wjb,pap,gcg,pair,gptfuzz,pez}
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python3 -u $H/acts_multi.py $MODEL $INP $OUT $SETS $LAYERS $g 4 $EXTRA > $OUT.log.g$g 2>&1 & done; wait
grep -h "rows in\|Error\|Traceback" $OUT.log.g* | cut -c1-200
python3 -u $H/acts_multi.py --merge $OUT $SETS $LAYERS 4
