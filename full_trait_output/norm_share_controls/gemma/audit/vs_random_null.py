#!/usr/bin/env python3
"""Are the refusal / jailbreak-direction transfer AUCs above chance? Null = folded AUC of 1000 random Gaussian unit
directions (and of 240 random plain Alpaca activations), per transfer family, Gemma and Llama. Prints each
direction's folded AUC with its percentile in the Gaussian null, and the null's mean / 95th percentile."""
import sys, numpy as np, torch
from pathlib import Path
NSC = Path.home() / "persona-vector-jbing/full_trait_output/norm_share_controls"
fa = lambda y, S: np.maximum(*(lambda a: (a, 1 - a))(np.array([__import__("sklearn.metrics").metrics.roc_auc_score(y, s) for s in S.T])))
def run(model):
    g = {}
    if model == "gemma":
        exec(open(NSC / "gemma/preamble_gemma.py").read(), g)
    else:
        exec(compile(open(NSC / "persona_part_test.py").read().split("splits = ")[0], str(NSC / "persona_part_test.py"), "exec"), {**g, "__file__": str(NSC / "persona_part_test.py")} | g); g = {**g}
    return g
for model in ("llama", "gemma"):
    g = {"__file__": str(NSC / "persona_part_test.py")}
    src = NSC / ("gemma/preamble_gemma.py" if model == "gemma" else "persona_part_test.py")
    code = open(src).read() if model == "gemma" else open(src).read().split("splits = ")[0]
    exec(compile(code, str(src), "exec"), g)
    np_ = g["np"]; T, Xh, yh, fam, ref, cb = g["T"], g["Xh"], g["yh"], g["fam"], g["ref"], g["cb"]
    unit = lambda v: v / np.linalg.norm(v); Qp = np.linalg.qr(T.T)[0]
    dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0)); r = unit(-ref)
    D = {"refusal": r, "refusal, persona removed": unit(r - Qp @ (Qp.T @ r)),
         "dmu": dmu, "dmu, persona removed (w_out)": unit(dmu - Qp @ (Qp.T @ dmu))}
    G = np.random.default_rng(0).standard_normal((1000, Xh.shape[1])); G /= np.linalg.norm(G, axis=1, keepdims=True)
    A = cb["acts_d0"].astype(np.float64); A /= np.linalg.norm(A, axis=1, keepdims=True)
    print(f"\n===== {model}  (cells: folded AUC, [percentile among 1000 random directions])")
    hdr = list(fam) + ["mean5"]; print(f"{'':30s}" + "".join(f"{h:>15s}" for h in hdr))
    nullG = {f: fa(y, X @ G.T) for f, (X, y) in fam.items()}; nullG["mean5"] = np.mean([nullG[f] for f in fam], 0)
    nullA = {f: fa(y, X @ A.T) for f, (X, y) in fam.items()}; nullA["mean5"] = np.mean([nullA[f] for f in fam], 0)
    print(f"{'random dir: mean / 95th pct':30s}" + "".join(f"{nullG[h].mean():8.3f}/{np.percentile(nullG[h], 95):.3f}" for h in hdr))
    print(f"{'random Alpaca act: mean / 95th':30s}" + "".join(f"{nullA[h].mean():8.3f}/{np.percentile(nullA[h], 95):.3f}" for h in hdr))
    for n, v in D.items():
        a = {f: fa(y, (X @ v)[:, None])[0] for f, (X, y) in fam.items()}; a["mean5"] = np.mean(list(a.values()))
        print(f"{n:30s}" + "".join(f"{a[h]:8.3f} [{100 * np.mean(nullG[h] < a[h]):3.0f}]" for h in hdr))
    print("n per family:", {f: (len(y), int(y.sum())) for f, (X, y) in fam.items()})
