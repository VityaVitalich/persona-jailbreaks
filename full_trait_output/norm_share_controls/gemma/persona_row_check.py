# GEMMA PORT of norm_share_controls/persona_row_check.py (gemma/port.py); model-specific inputs swapped, logic unchanged.
#!/usr/bin/env python3
"""PersonaJB row for the appendix controls table, through the same detector() code path and 50 pool splits as
remove_persona_all_controls.py (should reproduce the published 0.854 / 0.760). Writes persona_row_check.json."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(compile(open(Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py")).read(), "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py", "exec"))
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]
r = detector(T); print(f"PersonaJB traits: HB {r[0]:.3f} transfer {r[1]:.3f}")
json.dump({"hb": r[0], "transfer": r[1]}, open(HERE / "persona_row_check.json", "w"))
