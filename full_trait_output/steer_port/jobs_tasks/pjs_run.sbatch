#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/steer_port/logs/%x-%j.out
# One node, up to 4 single-GPU runner processes. TASKFILE has one line per GPU: "<runner .py> <args...>"
# (relative to steer_port). Runners are resume-safe, so the same TASKFILE can be resubmitted after a timeout.
# Copies may be queued on several partitions / accounts; a per-run lock lets only one of them work at a time.
#   sbatch -J <name> --environment=/users/vvmoskvoretskii/MR-Eval/container/serving.toml jobs/pjs_run.sbatch jobs/tasks/<file>
set -uo pipefail
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/steer_port; cd $P
TASKFILE=$1; KIND=${2:-llama}     # llama | olmo | gemma
GATE=${GATE:-gates/smoke_ok}
[ "${NOGATE:-0}" = 1 ] || [ -f $GATE ] || { echo "gate $GATE missing (smoke not accepted), exiting"; exit 0; }
mkdir -p locks; LOCK=locks/$SLURM_JOB_NAME
mkdir $LOCK 2>/dev/null || { echo "lock $LOCK held by $(cat $LOCK/owner 2>/dev/null), exiting"; exit 0; }
echo "$SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)" > $LOCK/owner
trap 'mv $LOCK $LOCK.released.$SLURM_JOB_ID' EXIT     # released (renamed, not removed) so a resubmit can run
export HF_HOME=$U/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
if [ "$KIND" = gemma ]; then export PYTHONPATH=$U/pyoverlay/gemma4_tf58:$U/pyoverlay/steer_py312 CHUNK_Q=512
else export PYTHONPATH=$U/pyoverlay/steer_py312; fi
set -a; . $P/openai.env; set +a
python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))"
LLAMA=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-8B-Instruct
python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$LLAMA'); assert 'Cutting Knowledge Date' in t.chat_template; print('official template ok')" || exit 1
export LLAMA OLMO=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('allenai/Olmo-3-7B-Instruct', local_files_only=True))") \
       GEMMA=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('google/gemma-4-31B-it', local_files_only=True))")
mkdir -p logs/run
g=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue; [ "${line:0:1}" = "#" ] && continue
  tag=$(basename $TASKFILE .txt)_g$g
  echo "GPU $g: $line"
  ( eval "CUDA_VISIBLE_DEVICES=$g python3 -u $line" ) > logs/run/${tag}_$SLURM_JOB_ID.log 2>&1 &
  g=$((g+1))
done < $TASKFILE
wait
for f in logs/run/$(basename $TASKFILE .txt)_g*_$SLURM_JOB_ID.log; do echo "== $f"; grep -h "ASR = \|Traceback\|Error" $f | tail -8; done
echo ALL_DONE $(date -u +%FT%TZ)
