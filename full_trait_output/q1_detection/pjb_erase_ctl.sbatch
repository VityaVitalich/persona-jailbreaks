#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/logs/%x-%j.out
# Persona erasure test (q1_erase_ctl.py), CPU only.
#   sbatch -A a0265 -p normal -J pjs_erase_<model> --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml pjb_erase.sbatch MODEL [--smoke]
Q=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection; cd $Q
mkdir -p locks; mkdir locks/$SLURM_JOB_NAME 2>/dev/null || { echo "lock held, exiting"; exit 0; }
[ -n "${2:-}" ] && [ -f out/erase_ctl_$1_${2#--cell=}.json ] && { echo "output exists, exiting"; mv locks/$SLURM_JOB_NAME locks/$SLURM_JOB_NAME.released.$SLURM_JOB_ID; exit 0; }
echo "$SLURM_JOB_ID" > locks/$SLURM_JOB_NAME/owner; trap 'mv locks/$SLURM_JOB_NAME locks/$SLURM_JOB_NAME.released.$SLURM_JOB_ID' EXIT
export OMP_NUM_THREADS=72 OPENBLAS_NUM_THREADS=72 MKL_NUM_THREADS=72 PYTHONUNBUFFERED=1
python3 -c "import sklearn, joblib, numpy; print('sklearn', sklearn.__version__, 'joblib', joblib.__version__, 'numpy', numpy.__version__)"
python3 -u q1_erase_ctl.py $1 $Q/configs/q1_$1_hbwjb.json $Q/out/q1_$1_hbwjb.json $Q/out ${NJ:-120} ${2:-}
