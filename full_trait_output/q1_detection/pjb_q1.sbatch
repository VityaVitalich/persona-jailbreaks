#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/logs/%x-%j.out
# Q1 detector sweep (q1_sweep.py) then persona-removed-after-training scoring (q1_posthoc.py), CPU only, for each TAG in turn.
#   sbatch --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_q1.sbatch "gemma_hb gemma_hbwjb"
Q=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection; cd $Q
for t in $1; do
  [ -s $Q/out/q1_$t.json ] || OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -u q1_sweep.py $t $Q/configs/q1_$t.json $Q/out 140 | grep -v "^done L"
  [ -s $Q/out/q1_posthoc_$t.json ] || OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 python3 -u q1_posthoc.py $t $Q/configs/q1_$t.json $Q/out/q1_$t.json $Q/out
  echo "$(date -u +%FT%TZ) $t done: $(ls $Q/out/q1_$t.json $Q/out/q1_posthoc_$t.json 2>/dev/null | wc -l)/2 files"
done
