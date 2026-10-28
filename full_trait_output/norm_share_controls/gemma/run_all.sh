#!/bin/bash
# Gemma ports of the non-steering norm_share_controls analyses, in dependency order; one log per script.
cd "$(dirname "$0")"; mkdir -p logs
for f in remove_persona_all_controls remove_nonpersona_span w_out_in_spans appendix_controls_data persona_part_pervector persona_part_spans refdmu_tables what_is_dmu_out template_test/dilution_pressure template_test/unrestricted_persona probe/max_activating; do
  echo "=== $f $(date +%T)"; python3.13 $f.py > logs/$(basename $f).log 2>&1 && echo "ok $f" || { echo "FAIL $f"; tail -5 logs/$(basename $f).log; }
done; echo ALL_DONE
