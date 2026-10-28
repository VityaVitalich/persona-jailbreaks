#!/usr/bin/env python3
"""Refusal and jailbreak (dmu) directions on Gemma: remove the persona component vs other 240-dim spans.
Spans: trait span; Alpaca / No Robots / Wikipedia activation spans (draw 4) as is and with their persona component
removed (-P, orthogonal to the trait span); random Gaussian. HB = folded AUC on the test pool (47 splits with both
classes; dmu re-estimated on each split's train rows); transfer = mean folded AUC over 5 families (refusal fixed; dmu
from all HB rows). CI: refusal transfer difference vs intact, paired bootstrap over prompts within each family (1000)."""
import numpy as np
from pathlib import Path
exec(open(Path.home() / "persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py").read())
from sklearn.metrics import roc_auc_score
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]; unit = lambda v: v / np.linalg.norm(v)
Qp = qr(T); clean = lambda B: qr(unit_rows(B.astype(np.float64) - (B.astype(np.float64) @ Qp) @ Qp.T))
SP = {"none (intact)": None, "persona": Qp, "Alpaca acts": qr(cb["acts_d4"]), "Alpaca acts - P": clean(cb["acts_d4"]),
      "No Robots acts - P": clean(eb["nr_d4"]), "Wikipedia acts - P": clean(eb["wiki_d4"]), "random": qr(cb["gauss_d4"])}
rm = lambda v, Q: unit(v) if Q is None else unit(v - Q @ (Q.T @ v))
sp = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]; sp = [(a, b) for a, b in sp if 0 < yh[b].sum() < len(b)]
fa = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
r = -ref  # oriented like the Llama analyses (refusal = -Arditi); folded AUC is sign-free anyway
dmu_all = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))
dmu_tr = lambda tr: Xh[tr][yh[tr] == 1].mean(0) - Xh[tr][yh[tr] == 0].mean(0)
rng = np.random.default_rng(0); boot = {f: [rng.integers(0, len(y), len(y)) for _ in range(1000)] for f, (X, y) in fam.items()}
def xf_boot(v):  # (1000,) bootstrap of mean5 transfer
    return np.mean([[fa(y[i], (X @ v)[i]) for i in boot[f]] for f, (X, y) in fam.items()], 0)
base_r = xf_boot(rm(r, None)); base_d = xf_boot(rm(dmu_all, None))
print(f"{'span removed':20s} | {'share':>5s} {'HB':>6s} {'xfer':>6s} {'d xfer [95% CI]':>22s} || {'share':>5s} {'HB':>6s} {'xfer':>6s} {'d xfer [95% CI]':>22s}")
print(f"{'':20s} | {'refusal':^45s} || {'jailbreak direction (dmu)':^45s}")
for n, Q in SP.items():
    out = []
    for v, hbf, base in ((r, lambda tr: r, base_r), (dmu_all, dmu_tr, base_d)):
        sh = 0.0 if Q is None else float(np.sum((Q.T @ unit(v)) ** 2))
        hb_ = np.mean([fa(yh[te], Xh[te] @ rm(hbf(tr), Q)) for tr, te in sp])
        b = xf_boot(rm(v, Q)); d = b - base
        out.append(f"{sh:5.3f} {hb_:6.3f} {np.mean(b):6.3f} {np.mean(d):+7.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
    print(f"{n:20s} | " + " || ".join(out), flush=True)
