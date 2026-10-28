#!/bin/bash
# all four attacks on 2 behaviours with few steps, sequentially in one debug allocation
P=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/qwen3_32b
for m in GCG PEZ GPTFuzz PAIR; do ATTACKS=$P/attacks_smoke CFGDIR=$P/configs_smoke timeout 1200 bash $P/scripts/pjb_attack.sbatch $m $P/scripts/ids_smoke.txt; done
