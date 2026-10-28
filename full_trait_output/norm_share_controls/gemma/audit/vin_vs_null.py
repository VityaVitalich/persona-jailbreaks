#!/usr/bin/env python3
"""Refusal and jailbreak direction (dmu, all HB rows): full, persona component v_in, remainder v_out.
Per model: HarmBench (all human-jailbreak rows, folded) and transfer (mean of 5 families, folded) AUC;
  pct_G   percentile among 1000 random Gaussian unit directions (full space)
  pct_P   percentile among 1000 random unit directions inside the trait span (Gaussian combination of the span basis)
  d_G     AUC minus the MEAN random-Gaussian AUC, 95% CI = bootstrap over prompts (1000, per family; the null mean is
          recomputed on each resample, so the CI covers sampling noise of the prompts only)."""
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
NSC = Path.home() / "persona-vector-jbing/full_trait_output/norm_share_controls"
def auc_many(y, S):
    from scipy.stats import rankdata
    R = np.apply_along_axis(rankdata, 0, S); n1 = y.sum(); n0 = len(y) - n1
    a = (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0); return np.maximum(a, 1 - a)
OUTJ = {}
for model in ("llama", "gemma"):
    src = NSC / ("gemma/preamble_gemma.py" if model == "gemma" else "persona_part_test.py")
    code = open(src).read() if model == "gemma" else open(src).read().split("splits = ")[0]
    g = {"__file__": str(src)}; exec(compile(code, str(src), "exec"), g)
    T, Xh, yh, fam, ref = g["T"], g["Xh"], g["yh"], g["fam"], g["ref"]
    unit = lambda v: v / np.linalg.norm(v); Qp = np.linalg.qr(T.T)[0]; d = Xh.shape[1]; rng = np.random.default_rng(0)
    G = rng.standard_normal((1000, d)); G /= np.linalg.norm(G, axis=1, keepdims=True)
    P = rng.standard_normal((1000, Qp.shape[1])) @ Qp.T; P /= np.linalg.norm(P, axis=1, keepdims=True)
    V = {}
    for n, v in (("refusal", unit(-ref)), ("dmu", unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0)))):
        pin = Qp @ (Qp.T @ v); V[n] = v; V[f"{n} v_in"] = unit(pin); V[f"{n} v_out"] = unit(v - pin)
    names = list(V); W = np.stack([V[n] for n in names])
    sets = {"HB": (Xh, yh), **fam}
    S = {k: (X @ W.T, X @ G.T, X @ P.T) for k, (X, y) in sets.items()}
    def stats(idx=None):  # -> dict set -> (auc dirs, aucs G, aucs P)
        out = {}
        for k, (X, y) in sets.items():
            i = slice(None) if idx is None else idx[k]; a, b, c = S[k]
            out[k] = (auc_many(y[i], a[i]), auc_many(y[i], b[i]), auc_many(y[i], c[i]) if idx is None else None)
        return out
    st = stats(); xf = lambda st, j: np.mean([st[f][j] for f in fam], 0)
    boot_d = {"HB": [], "xf": []}
    for _ in range(300):
        idx = {k: rng.integers(0, len(y), len(y)) for k, (X, y) in sets.items()}; b = stats(idx)
        boot_d["HB"].append(b["HB"][0] - b["HB"][1].mean()); boot_d["xf"].append(xf(b, 0) - xf(b, 1).mean())
    print(f"\n===== {model}   random Gaussian dir: HB mean {st['HB'][1].mean():.3f}, transfer mean {xf(st, 1).mean():.3f} (95th {np.percentile(xf(st, 1), 95):.3f}) | "
          f"random dir in trait span: HB mean {st['HB'][2].mean():.3f}, transfer mean {xf(st, 2).mean():.3f} (95th {np.percentile(xf(st, 2), 95):.3f})")
    print(f"{'':16s} | {'HB AUC':>6s} {'pctG':>4s} {'pctP':>4s} {'d vs mean random [CI]':>24s} | {'xfer':>6s} {'pctG':>4s} {'pctP':>4s} {'d vs mean random [CI]':>24s}")
    for j, n in enumerate(names):
        row = []
        for key, a, nG, nP in (("HB", st["HB"][0][j], st["HB"][1], st["HB"][2]), ("xf", xf(st, 0)[j], xf(st, 1), xf(st, 2))):
            bd = np.array(boot_d[key])[:, j]
            row.append(f"{a:6.3f} {100 * np.mean(nG < a):4.0f} {100 * np.mean(nP < a):4.0f} {a - nG.mean():+8.3f} [{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]")
        print(f"{n:16s} | " + " | ".join(row))
        bdx = np.array(boot_d['xf'])[:, j]; OUTJ.setdefault(model, {})[n] = dict(transfer=float(xf(st, 0)[j]), d=float(xf(st, 0)[j] - xf(st, 1).mean()), lo=float(np.percentile(bdx, 2.5)), hi=float(np.percentile(bdx, 97.5)), pctG=float(100 * np.mean(xf(st, 1) < xf(st, 0)[j])), pctP=float(100 * np.mean(xf(st, 2) < xf(st, 0)[j])))
    OUTJ[model]['_null'] = dict(mean=float(xf(st, 1).mean()), p95=float(np.percentile(xf(st, 1), 95)), mean_in_span=float(xf(st, 2).mean()))

import json; json.dump(OUTJ, open(Path(__file__).resolve().parent / 'vin_vs_null.json', 'w'), indent=1)
