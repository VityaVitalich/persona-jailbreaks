#!/usr/bin/env python3
"""50-split rerun of the Alpaca PCA-229 row of pcs_persona_component.py (same 50 pool splits as PersonaJB).
Writes pcs_persona_component_50.json."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(compile(open(HERE / "pcs_persona_component.py").read().split("res = {}")[0], str(HERE / "pcs_persona_component.py"), "exec"))
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]
P = unit_rows(cb["alpaca_pca"].astype(np.float64)); res = {"whole": detector(P)}
print(f"Alpaca PCA-229, 50 splits: whole HB {res['whole'][0]:.3f} transfer {res['whole'][1]:.3f}")
for rn in ("persona traits", "No Robots acts", "random Gaussian"):
    Q = REM[rn][0]; res[rn] = detector(unit_rows(P - P @ Q @ Q.T))
    print(f"  {rn:16s} removed: HB {res[rn][0]:.3f} transfer {res[rn][1]:.3f}")
json.dump(res, open(HERE / "pcs_persona_component_50.json", "w"), indent=1)
