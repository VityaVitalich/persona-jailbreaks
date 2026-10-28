#!/bin/bash
#SBATCH --account=ab023
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/qwen3_32b/probe_methods/logs/%x-%j.out
# CPU-only: probe-method comparison (probe_methods.py) on Qwen L48 sweep activations + Llama L16 shared data.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_probe_methods.sbatch
set -uo pipefail
P=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/qwen3_32b; D=$P/probe_methods
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u $D/probe_methods.py $P/sweep $P/transfer $D/llama $D/out 64 > $D/logs/run.log 2>&1
echo "$(date -u +%FT%TZ) probe_methods rc=$?" >> $D/STATUS
