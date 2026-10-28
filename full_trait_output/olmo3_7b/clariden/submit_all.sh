#!/bin/bash
# Full OLMo-3-7B attack fan-out (run with bash on the clariden login node): GCG 159 behaviours over 5 nodes, PEZ 400 on
# 1 node, GPTFuzz 400 over 2 nodes, PAIR 400 over 8 nodes. ACCOUNT overrides ab023 (fallback a0265).
set -euo pipefail
S=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/olmo3_7b/scripts; C=/users/vvmoskvoretskii/MR-Eval/container
ACC=${ACCOUNT:-ab023}; I=$S/ids; mkdir -p $I; cd $S
split -n l/5 -d ids_gcg.txt $I/gcg_; split -n l/2 -d ids_all.txt $I/gptfuzz_; split -n l/8 -d ids_all.txt $I/pair_; cp ids_all.txt $I/pez_00
for f in $I/gcg_*; do sbatch --account=$ACC --environment=$C/harmbench.toml -J olmo-$(basename $f) pjb_attack.sbatch GCG $f; done
sbatch --account=$ACC --environment=$C/harmbench.toml -J olmo-pez_00 pjb_attack.sbatch PEZ $I/pez_00
for f in $I/gptfuzz_*; do sbatch --account=$ACC -J olmo-$(basename $f) pjb_served.sbatch GPTFuzz $f; done
for f in $I/pair_*; do sbatch --account=$ACC -J olmo-$(basename $f) pjb_served.sbatch PAIR $f; done
