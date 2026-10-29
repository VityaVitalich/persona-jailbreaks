#!/bin/bash
# Runs on the clariden login node (tmux q1drive): submit the q1 debug sweeps one by one (debug-qos: 2 submitted / 1
# running per user) and log until all 8 posthoc files exist. Skips a group whose job is already queued or done.
Q=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection; cd $Q
E=--environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml
queue=("llama:llama_hbwjb llama_hb" "olmo:olmo_hbwjb olmo_hb"); i=0
while true; do
  if [ "$i" -lt "${#queue[@]}" ]; then
    nm=q1-${queue[$i]%%:*}; tags=${queue[$i]#*:}
    if sacct -u $USER -S 2026-10-28T15:30 -X -n -o JobName%14,State | grep -q "^ *$nm .*\(PENDING\|RUNNING\|COMPLETED\)"; then i=$((i+1)); continue; fi
    if [ "$(squeue -u $USER -p debug -h | wc -l)" -lt 2 ]; then
      j=$(sbatch -p debug $E -J $nm pjb_q1.sbatch "$tags" 2>&1 | grep -o 'job [0-9]*'); echo "$(date +%H:%M) submitted $nm ($tags) -> ${j:-FAILED}"; [ -n "$j" ] && i=$((i+1))
    fi
  fi
  d=$(ls out/ | grep -c '^q1_posthoc_.*_hb.*\.json$')
  echo "$(date +%H:%M) posthoc files $d/8 | $(sacct -u $USER -S 2026-10-28T15:30 -X -n -o JobID,JobName%14,State,Elapsed | grep q1- | tr -s ' ' | tr '\n' ';')"
  [ "$d" -ge 8 ] && { echo ALL_DONE; break; }
  sleep 300
done
