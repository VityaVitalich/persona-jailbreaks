#!/usr/bin/env python3
"""Transfer version of the single-vector part of persona_part_test.py: median over the 229 vectors of the
folded AUC of score = h . v on each transfer family, averaged over the 5 families. Writes persona_part_transfer.json."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = open(HERE / "persona_part_test.py").read().split("res = {}")[0]
exec(compile(src, str(HERE / "persona_part_test.py"), "exec"))
xs = lambda V: float(np.mean([np.median(fold(rank_auc(y, X @ V.T))) for X, y in fam.values()]))
fx = lambda V: {f: float(np.median(fold(rank_auc(y, X @ V.T)))) for f, (X, y) in fam.items()}
res = {}
for dn, V in DICTS.items():
    V = unit_rows(V.astype(np.float64))
    print(f"\n=== {dn}: whole vector v -> transfer {xs(V):.3f}   per family " + " ".join(f"{k} {v:.2f}" for k, v in fx(V).items()))
    print(f"{'span':20s} {'v_in':>6s} {'v_in-ref':>9s} {'v_out':>6s}")
    for sn, B in SPANS.items():
        Q, _ = np.linalg.qr(B.T); Vin = V @ Q @ Q.T; Vout = V - Vin; Vnr = Vin - np.outer(Vin @ ref, ref)
        r = dict(v_in=xs(unit_rows(Vin)), v_in_noref=xs(unit_rows(Vnr)), v_out=xs(unit_rows(Vout)))
        res[f"{dn}|{sn}"] = r
        print(f"{sn:20s} {r['v_in']:6.3f} {r['v_in_noref']:9.3f} {r['v_out']:6.3f}")
print(f"\nrefusal direction alone: transfer {np.mean([fold(rank_auc(y, (X @ ref)[:, None]))[0] for X, y in fam.values()]):.3f}")
G = unit_rows(np.random.default_rng(5).standard_normal((229, 4096)))
print(f"random Gaussian single directions (floor): transfer {xs(G):.3f}")
json.dump(res, open(HERE / "persona_part_transfer.json", "w"), indent=1)
