#!/bin/bash
#SBATCH --account=ab023
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:20:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/olmo3_7b/logs/%x-%j.out
# OLMo-3-7B smoke on the debug partition. Submit WITHOUT --environment.
#   sbatch -J olmo-smoke-a smoke.sbatch a   compat check (harmbench image vs native) + GCG + PEZ (few steps, 2 behaviours)
#   sbatch -J olmo-smoke-b smoke.sbatch b   GPTFuzz + PAIR (MS=PAIR: PAIR only) (few steps) + gen/acts/judge on 4-8 prompts per set
set -uo pipefail
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb/olmo3_7b; S=$P/scripts; HB=$U/MR-Eval-hbattacks/harmbench
export HF_HOME=$U/hf_cache; C=/users/vvmoskvoretskii/MR-Eval/container; export ATTACKS=$P/attacks_smoke CFGDIR=$P/configs_smoke; mkdir -p $P/smoke $ATTACKS/logs
if [ "$1" = a ]; then
  srun --overlap -n1 --gres=gpu:4 --environment=$C/serving.toml bash -c "export HF_HOME=$U/hf_cache; CUDA_VISIBLE_DEVICES=0 python3 $S/compat_check.py $P/smoke/native.pt"
  srun --overlap -n1 --gres=gpu:4 --environment=$C/harmbench.toml bash -c "export HF_HOME=$U/hf_cache; cd $HB; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$HB python3 $S/compat_check.py $P/smoke/compat.pt; python3 $S/compat_check.py --compare $P/smoke/compat.pt $P/smoke/native.pt"
  for m in GCG PEZ; do srun --overlap -n1 --gres=gpu:4 --environment=$C/harmbench.toml timeout 1200 bash $S/pjb_attack.sbatch $m $S/ids_smoke.txt; done
else
  for m in ${MS:-GPTFuzz PAIR}; do timeout 1800 bash $S/pjb_served.sbatch $m $S/ids_smoke.txt; done
  srun --overlap -n1 --gres=gpu:4 --environment=$C/serving.toml bash -c "export TRANSFER=$P/transfer_smoke TEST=--test; bash $S/pjb_transfer.sbatch hb,wjb,pap,gcg,pair,pez,gptfuzz"  # the image's env is /root/.local/bin/env (not executable)
fi
tail -5 $ATTACKS/logs/STATUS
