#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/logs/%x-%j.out
# Q1 detector sweep (q1_sweep.py), CPU only.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_q1.sbatch MODEL
Q=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection; cd $Q
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -u q1_sweep.py $1 $Q/q1_$1_config.json $Q/out 140
