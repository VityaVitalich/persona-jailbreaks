#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/hb_select/logs/%x-%j.out
# Readout selection on the HB validation split, PersonaJB vs raw (CPU only). MODELS: space-separated; PREP=1 builds inputs.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_hb_select.sbatch "olmo qwen"
H=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/hb_select; cd $H
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
[ "${PREP:-0}" = 1 ] && python3 -u hb_select_prep.py
for m in $1; do python3 -u hb_select.py $m $H/hb_select_$m.json $H/out 140 ${CVFLAG:-} 2>&1 | grep -v "Parallel(n_jobs\|Done .* tasks"; done
