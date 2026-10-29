#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/logs/%x-%j.out
# Q1 option A: persona removed after training (q1_posthoc.py), CPU only.
#   sbatch -p debug --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_q1_posthoc.sbatch MODEL
Q=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection; cd $Q
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
python3 -u q1_posthoc.py $1 $Q/q1_$1_config.json $Q/out/q1_$1.json $Q/out
