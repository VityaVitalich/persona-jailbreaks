# GEMMA PORT of norm_share_controls/w_out_in_spans.py (gemma/port.py); model-specific inputs swapped, logic unchanged.
#!/usr/bin/env python3
"""
Where does w_out (the jailbreak direction with its persona component removed) lie? If ordinary activation spans do
not contain it, dictionaries built from ordinary text can only reach the jailbreak signal through their persona
component, which would explain why they lose transfer without persona while the jailbreak direction does not.

Directions: w_out, w_in (persona component of the jailbreak direction), the jailbreak direction (dmu, all HarmBench
human-jailbreak rows), the refusal direction (-Arditi) and the refusal direction with its persona component removed.
Spans (240 dims unless noted): trait span; Wikipedia / No Robots / Alpaca activations (draw 0) as is and with their
persona component removed (draw 4, as in remove_nonpersona_span.py); Alpaca PCA-240; HarmBench PCA-240 (top variance
of the jailbreak prompts themselves, no labels); random Gaussian; and the union of 5 draws of Alpaca activations
(1200 dims) against a random 1200-dim span.
  share  squared norm of the direction inside the span
  AUC    for w_out only: folded AUC of the score h.(unit projection of w_out onto the span), HarmBench (in-sample,
         all rows) and mean over the 5 transfer families
Writes w_out_in_spans.json.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(compile(open(Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py")).read(), "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py", "exec"))
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]
unit = lambda v: v / np.linalg.norm(v)
Qp = qr(T)
clean = lambda B: qr(unit_rows(B.astype(np.float64) - (B.astype(np.float64) @ Qp) @ Qp.T))
dmu = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0)); pin = Qp @ (Qp.T @ dmu)
r = -ref
D = {"w_out": unit(dmu - pin), "w_in": unit(pin), "jailbreak direction": dmu, "refusal direction": r,
     "refusal, persona removed": unit(r - Qp @ (Qp.T @ r))}
rng = np.random.default_rng(0)
SP = {"trait span": Qp, "Wikipedia acts": qr(eb["wiki_d0"]), "No Robots acts": qr(eb["nr_d0"]), "Alpaca acts": qr(cb["acts_d0"]),
      "Wikipedia acts - P": clean(eb["wiki_d4"]), "No Robots acts - P": clean(eb["nr_d4"]), "Alpaca acts - P": clean(cb["acts_d4"]),
      "Alpaca PCA-240": qr(cb["alpaca_pca"]), "HarmBench PCA-240": qr(cb["hb_pca"]), "random Gaussian": qr(cb["gauss_d0"]),
      "Alpaca acts x5 (1200)": qr(np.vstack([cb[f"acts_d{i}"] for i in range(5)])), "random (1200)": qr(rng.standard_normal((1200, 5376)))}
auc_hb = lambda v: float(fold(rank_auc(yh, (Xh @ v)[:, None]))[0])
auc_xf = lambda v: float(np.mean([fold(rank_auc(y, (X @ v)[:, None]))[0] for X, y in fam.values()]))
res = {"share": {}, "w_out_projection_auc": {}}
print(f"{'span':24s} " + "".join(f"{d[:18]:>20s}" for d in D))
for sn, Q in SP.items():
    res["share"][sn] = {d: float(np.sum((Q.T @ v) ** 2)) for d, v in D.items()}
    print(f"{sn:24s} " + "".join(f"{res['share'][sn][d]:20.3f}" for d in D))
w = D["w_out"]
print(f"\nw_out projected onto each span: folded AUC (HB in-sample, transfer)   [w_out itself: {auc_hb(w):.3f} {auc_xf(w):.3f}]")
for sn, Q in SP.items():
    if sn == "trait span": continue
    p = unit(Q @ (Q.T @ w)); res["w_out_projection_auc"][sn] = dict(hb=auc_hb(p), transfer=auc_xf(p))
    print(f"  {sn:24s} {auc_hb(p):.3f} {auc_xf(p):.3f}")
res["w_out_projection_auc"]["w_out itself"] = dict(hb=auc_hb(w), transfer=auc_xf(w))
json.dump(res, open(HERE / "w_out_in_spans.json", "w"), indent=1)
