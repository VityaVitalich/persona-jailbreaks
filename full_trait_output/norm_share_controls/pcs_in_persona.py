#!/usr/bin/env python3
"""
Are the Alpaca (and HarmBench) principal components persona-like?
(1) share of each PC's squared norm inside the persona span vs inside 229-dim spans of Wikipedia / No Robots
    activations (5 draws) and random Gaussian; for PCs 1-10, 11-50, 51-229.
(2) subspace overlap: mean squared principal cosine between the 229-dim spans, and #principal cosines > 0.9.
(3) the predictive PCs (top 20 by HarmBench folded AUC): transfer AUC of the PC, its persona part, the rest.
(4) the 3 traits most aligned with each of those PCs.
Writes pcs_in_persona.json.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(compile(open(HERE / "persona_part_pervector.py").read().split("TOP = 50")[0], str(HERE / "persona_part_pervector.py"), "exec"))
names = json.load(open(OUT / "trait_names_layer16.json"))
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]
Qp = qr(T)
CTRL = {"Wikipedia acts": [qr(eb[f"wiki_d{i}"]) for i in range(5)], "No Robots acts": [qr(eb[f"nr_d{i}"]) for i in range(5)],
        "Random Gaussian": [qr(cb[f"gauss_d{i}"]) for i in range(5)]}
share = lambda Q, V: np.sum((V @ Q) ** 2, 1)
res = {}
for pn in ("alpaca_pca", "hb_pca"):
    P = unit_rows(cb[pn].astype(np.float64))
    print(f"\n===== {pn}: share of each PC inside a 229-dim span (mean over the PCs in the range)")
    rows = {"Persona traits": share(Qp, P), **{k: np.mean([share(Q, P) for Q in Qs], 0) for k, Qs in CTRL.items()}}
    print(f"{'span':18s} {'PC1-10':>7s} {'PC11-50':>8s} {'PC51-229':>9s} | overlap of spans: mean cos^2, #cos>0.9")
    Qpc = qr(P)
    for k, s in rows.items():
        Qs = [Qp] if k == "Persona traits" else CTRL[k]
        sv_ = [np.linalg.svd(Qpc.T @ Q, compute_uv=False) for Q in Qs]
        print(f"{k:18s} {s[:10].mean():7.3f} {s[10:50].mean():8.3f} {s[50:].mean():9.3f} | "
              f"{np.mean([np.mean(x ** 2) for x in sv_]):15.3f} {np.mean([np.sum(x > 0.9) for x in sv_]):6.1f}")
    a_hb = hbA(P); top = np.argsort(-a_hb)[:20]
    Qin = Qp
    Vin, Vout = P @ Qin @ Qin.T, P - P @ Qin @ Qin.T
    x, xi, xo = xfA(P), xfA(unit_rows(Vin)), xfA(unit_rows(Vout))
    print(f"top-20 predictive PCs (by HarmBench AUC): transfer whole {x[top].mean():.3f}, persona part {xi[top].mean():.3f}, rest {xo[top].mean():.3f}; "
          f"persona part >= whole in {int(np.sum(xi[top] >= x[top]))}/20, rest <= noise bar {p95:.3f} in {int(np.sum(xo[top] <= p95))}/20")
    print(f"{'PC':>4s} {'HB AUC':>6s} {'xfer':>5s} {'in persona':>10s} | most aligned traits (cosine)")
    for i in top[:10]:
        c = T @ P[i]; o = np.argsort(-np.abs(c))[:3]
        print(f"{i + 1:4d} {a_hb[i]:6.3f} {x[i]:5.3f} {rows['Persona traits'][i]:10.2f} | " + ", ".join(f"{names[j]} {c[j]:+.2f}" for j in o))
    res[pn] = {k: v.tolist() for k, v in rows.items()} | {"hb_auc": a_hb.tolist(), "xf": x.tolist(), "xf_in": xi.tolist(), "xf_out": xo.tolist()}
json.dump(res, open(HERE / "pcs_in_persona.json", "w"))
