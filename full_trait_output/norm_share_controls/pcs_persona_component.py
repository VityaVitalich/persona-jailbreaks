#!/usr/bin/env python3
"""
Are the Alpaca / HarmBench PCs effective BECAUSE of their persona component?
Each PC p is split into p_in (shadow on the persona span) and p_out = p - p_in.
(A) detector: PersonaJB recipe (StandardScaler + LR C=10, 10 pool splits) on the 229 PCs, vs on the 229 p_out
    (persona component removed). Controls: remove instead the component in a 229-dim span of Wikipedia activations,
    of No Robots activations, or of random Gaussian directions (share of squared norm removed reported).
(B) single PC: transfer AUC of h.p, h.p_in, h.p_out for the PCs whose own transfer AUC is above the random-direction
    95th percentile (the only PCs with a transfer signal to explain).
Folded AUC. Writes pcs_persona_component.json.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(compile(open(HERE / "persona_part_pervector.py").read().split("TOP = 50")[0], str(HERE / "persona_part_pervector.py"), "exec"))
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]
REM = {"persona traits": [qr(T)], "Wikipedia acts": [qr(eb[f"wiki_d{i}"]) for i in range(3)],
       "No Robots acts": [qr(eb[f"nr_d{i}"]) for i in range(3)], "random Gaussian": [qr(cb[f"gauss_d{i}"]) for i in range(3)]}
res = {}
for pn, label in (("alpaca_pca", "Alpaca PCA-229"), ("hb_pca", "HarmBench PCA-229")):
    P = unit_rows(cb[pn].astype(np.float64)); d0 = detector(P)
    print(f"\n===== {label}: detector on the 229 PCs: HarmBench {d0[0]:.3f}, transfer {d0[1]:.3f}")
    print(f"{'component removed':20s} {'share removed':>13s} | detector HB  transfer")
    res[pn] = {"whole": d0}
    for rn, Qs in REM.items():
        out = [(float(np.mean(np.sum((P @ Q) ** 2, 1))), *detector(unit_rows(P - P @ Q @ Q.T))) for Q in Qs]
        m = np.mean(out, 0); res[pn][rn] = out
        print(f"{rn:20s} {m[0]:13.3f} | {m[1]:11.3f} {m[2]:9.3f}")
    Q = qr(T); x, xi, xo = xfA(P), xfA(unit_rows(P @ Q @ Q.T)), xfA(unit_rows(P - P @ Q @ Q.T))
    s = np.where(x > p95)[0]
    print(f"single PCs with transfer above the noise bar {p95:.3f}: {len(s)} of 229")
    for i in s:
        print(f"   PC{i + 1:<4d} whole {x[i]:.3f}  persona part {xi[i]:.3f}  rest {xo[i]:.3f}")
    res[pn]["single"] = {"idx": s.tolist(), "whole": x[s].tolist(), "in": xi[s].tolist(), "out": xo[s].tolist()}
json.dump(res, open(HERE / "pcs_persona_component.json", "w"), indent=1)
