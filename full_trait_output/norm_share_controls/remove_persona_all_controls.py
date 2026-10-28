#!/usr/bin/env python3
"""
Remove the persona component from every control dictionary and rerun the PersonaJB detector (StandardScaler + LR
C=10, the same 50 pool splits), folded AUC on HarmBench test and mean over the 5 transfer families.
Each dictionary vector v -> v - (shadow of v on the persona span), renormalized.
Controls for the removal: remove instead the component in a 229-dim span of No Robots activations (draw d4, not
used as a dictionary below) or of random Gaussian directions (gauss_d4). Share of squared norm removed reported.
Dictionaries: Wikipedia / No Robots / Alpaca activations, shuffled-label contrasts, random Gaussian (draws 0-2),
Alpaca PCA-229, HarmBench PCA-229. Writes remove_persona_all_controls.json.
"""
import json
from pathlib import Path
from joblib import Parallel, delayed
HERE = Path(__file__).resolve().parent
exec(compile(open(HERE / "persona_part_test.py").read().split("res = {}")[0], str(HERE / "persona_part_test.py"), "exec"))
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]
REM = {"persona": qr(T), "No Robots span": qr(eb["nr_d4"]), "random span": qr(cb["gauss_d4"])}
FAMS_ = [("Wikipedia acts", "wiki_d", eb), ("No Robots acts", "nr_d", eb), ("Alpaca acts", "acts_d", cb),
         ("Shuffled contrasts", "noise_d", cb), ("Random Gaussian", "gauss_d", cb)]
jobs = [(n, i, src[f"{p}{i}"]) for n, p, src in FAMS_ for i in range(3)] + [("Alpaca PCA-229", 0, cb["alpaca_pca"]), ("HarmBench PCA-229", 0, cb["hb_pca"])]

def run(n, i, V):
    V = unit_rows(V.astype(np.float64)); out = {"whole": detector(V)}
    for rn, Q in REM.items():
        out[rn] = (float(np.mean(np.sum((V @ Q) ** 2, 1))), *detector(unit_rows(V - V @ Q @ Q.T)))
    return n, i, out
R = Parallel(n_jobs=8)(delayed(run)(*j) for j in jobs)
res = {}
for n, i, out in R: res.setdefault(n, []).append(out)
json.dump(res, open(HERE / "remove_persona_all_controls.json", "w"), indent=1)
print(f"{'dictionary':20s} | {'whole':>11s} | {'persona removed':>22s} | {'No Robots span removed':>22s} | {'random span removed':>22s}")
print(f"{'':20s} | {'HB   xfer':>11s} | {'share  HB   xfer':>22s} | {'share  HB   xfer':>22s} | {'share  HB   xfer':>22s}")
for n, outs in res.items():
    w = np.mean([o["whole"] for o in outs], 0)
    line = f"{n:20s} | {w[0]:.3f} {w[1]:.3f} |"
    for rn in REM:
        m = np.mean([o[rn] for o in outs], 0); line += f" {m[0]:9.3f} {m[1]:.3f} {m[2]:.3f} |"
    print(line)
