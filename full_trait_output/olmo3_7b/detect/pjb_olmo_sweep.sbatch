#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/olmo3_7b/logs/%x-%j.out
# OLMo-3-7B layer x position sweep, PersonaJB vs raw LR (CPU only).
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_olmo_sweep.sbatch
P=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/olmo3_7b
cd $P/detect && OLMO_TR=$P/transfer OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -u olmo_sweep_pjb.py $P/trait_vectors/all_traits_no_filter $P/detect/out 120
