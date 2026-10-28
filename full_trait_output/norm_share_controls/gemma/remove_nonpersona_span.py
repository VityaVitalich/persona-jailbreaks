# GEMMA PORT of norm_share_controls/remove_nonpersona_span.py (gemma/port.py); model-specific inputs swapped, logic unchanged.
#!/usr/bin/env python3
"""
Activation-derived, persona-free control for persona removal. A random Gaussian span is a weak control: it barely
overlaps the directions in which activations vary. Here the control spans are built from real activations and then
cleaned of persona: take 240 activations not used as dictionary vectors (draw 4) of Wikipedia, No Robots or Alpaca
text, remove their projection onto the trait span, and use the span of what is left (240 dims, orthogonal to the
trait span). Removing it from a dictionary removes activation structure that is not persona.

Removals (each vector v -> v minus its projection onto the span, renormalized; detector and splits as in
remove_persona_all_controls.py: StandardScaler + LR C=10, 50 pool splits, folded AUC, HB test + mean of 5 families)
  persona            the trait span
  Wikipedia          span of 240 Wikipedia activations (wiki_d4), as is
  Wikipedia - P      the same activations with their persona component removed
  No Robots - P      240 No Robots activations (nr_d4), persona component removed
  Alpaca - P         240 Alpaca activations (acts_d4), persona component removed
  random             240 random Gaussian directions (gauss_d4)
For each span also: share of the total variance of the HarmBench activations (centered) it holds, and share of the
squared norm of the jailbreak direction (mean jailbroken - mean refused). Writes remove_nonpersona_span.json.
"""
import json
from pathlib import Path
from joblib import Parallel, delayed
HERE = Path(__file__).resolve().parent
exec(compile(open(Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py")).read(), "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py", "exec"))
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]
Qp = qr(T)
clean = lambda B: qr(unit_rows(B.astype(np.float64) - (B.astype(np.float64) @ Qp) @ Qp.T))
REM = {"persona": Qp, "Wikipedia": qr(eb["wiki_d4"]), "Wikipedia - P": clean(eb["wiki_d4"]),
       "No Robots - P": clean(eb["nr_d4"]), "Alpaca - P": clean(cb["acts_d4"]), "random": qr(cb["gauss_d4"])}
Xc = Xh - Xh.mean(0); tot = float(np.sum(Xc ** 2))
dmu = Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0); dmu /= np.linalg.norm(dmu)
span_stats = {n: dict(var_share=float(np.sum((Xc @ Q) ** 2)) / tot, dmu_share=float(np.sum((Q.T @ dmu) ** 2)),
                      overlap_with_persona=float(np.sum((Qp.T @ Q) ** 2)) / Q.shape[1]) for n, Q in REM.items()}
print(f"{'span':16s} {'HB variance share':>17s} {'jailbreak-dir share':>19s} {'mean overlap w/ persona':>23s}")
for n, v in span_stats.items(): print(f"{n:16s} {v['var_share']:17.3f} {v['dmu_share']:19.3f} {v['overlap_with_persona']:23.3f}")

FAMS_ = [("Wikipedia acts", "wiki_d", eb), ("No Robots acts", "nr_d", eb), ("Alpaca acts", "acts_d", cb),
         ("Shuffled contrasts", "noise_d", cb), ("Random Gaussian", "gauss_d", cb)]
jobs = [(n, i, src[f"{p}{i}"]) for n, p, src in FAMS_ for i in range(3)] + [("Alpaca PCA-240", 0, cb["alpaca_pca"])]

def run(n, i, V):
    V = unit_rows(V.astype(np.float64)); out = {"whole": detector(V)}
    for rn, Q in REM.items():
        out[rn] = (float(np.mean(np.sum((V @ Q) ** 2, 1))), *detector(unit_rows(V - V @ Q @ Q.T)))
    return n, i, out
R = Parallel(n_jobs=8)(delayed(run)(*j) for j in jobs)
res = {"spans": span_stats, "dicts": {}}
for n, i, out in R: res["dicts"].setdefault(n, []).append(out)
json.dump(res, open(HERE / "remove_nonpersona_span.json", "w"), indent=1)
print(f"\n{'dictionary':20s} | {'whole':>6s} | " + " | ".join(f"{rn:>17s}" for rn in REM))
print(f"{'transfer AUC':20s} | {'':6s} | " + " | ".join(f"{'share   xfer':>17s}" for _ in REM))
for n, outs in res["dicts"].items():
    w = np.mean([o["whole"] for o in outs], 0)
    print(f"{n:20s} | {w[1]:6.3f} | " + " | ".join(f"{np.mean([o[rn][0] for o in outs]):8.3f} {np.mean([o[rn][2] for o in outs]):.3f}" for rn in REM))
print(f"\n{'HarmBench AUC':20s} | " + " | ".join(f"{rn:>13s}" for rn in ["whole", *REM]))
for n, outs in res["dicts"].items():
    print(f"{n:20s} | {np.mean([o['whole'][0] for o in outs]):13.3f} | " + " | ".join(f"{np.mean([o[rn][1] for o in outs]):13.3f}" for rn in REM))
