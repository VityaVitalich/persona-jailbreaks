#!/usr/bin/env python3
"""
Persona-only steering arms: does the persona component of a steering direction steer on its own? Each arm is the
projection of a direction onto the span of the 229 trait vectors, renormalized; the full direction and its
persona-removed version were run before (steer_controls_v2_layer16.pt, steer_nopersona_layer16.pt,
steer_refdmu_layer16.pt).
  refusal_personaonly     -(Arditi refusal direction), persona component only
  dmu_personaonly         jailbreak direction (mean jailbroken - mean refused), persona component only (= w_in)
  alpacajb_d2_personaonly PersonaJB detector refit on the persona components of the 229 Alpaca activations of
                          acts_d2; normal mapped back (lies inside the persona span)
  sel_pca_1_personaonly   top Alpaca PC (sel_pca_1), persona component only
All unit norm and oriented toward jailbroken on HarmBench (sign flipped if needed; printed). Prints the persona share
of the original, the cosine with it and the folded HarmBench / transfer AUC as h.v.
Writes steer_personaonly_layer16.pt (steer_fixed_direction.py format).
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(HERE))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
unit_rows = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
fold = lambda a: max(a, 1 - a)
cfg = S.CFG["llama"]
T = unit_rows(np.load(cfg["T"]).astype(np.float64)); Qp = np.linalg.qr(T.T)[0]
cb = np.load(HERE / "control_bases_llama.npz")
old = torch.load(HERE / "steer_controls_v2_layer16.pt", weights_only=False)["vectors"]
X, y, _ = S.build_matrix(S.load_jsonl(cfg["human"][0]), torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
X = X.astype(np.float64)
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
fam = {f: S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)[:2] for f, (rp, ap) in cfg["fams"].items()}
keep = lambda M: (M @ Qp) @ Qp.T

def det(V):  # detector normal on dictionary V (same recipe as build_steer_nopersona.py)
    m = S.model(0).fit(X @ V.T, y); return unit(V.T @ (m[-1].coef_[0] / m[0].scale_))
A = unit_rows(cb["acts_d2"].astype(np.float64))
src = {"refusal": old["refusal"].double().numpy(), "dmu": old["dmu"].double().numpy(),
       "alpacajb_d2": old["alpacajb_d2"].double().numpy(), "sel_pca_1": old["sel_pca_1"].double().numpy()}
vec = {"refusal_personaonly": unit(keep(src["refusal"][None])[0]), "dmu_personaonly": unit(keep(src["dmu"][None])[0]),
       "alpacajb_d2_personaonly": det(unit_rows(keep(A))), "sel_pca_1_personaonly": unit(keep(src["sel_pca_1"][None])[0])}
sgn_auc = lambda v, Xm, ym: rank_auc(np.asarray(ym), (np.asarray(Xm, np.float64) @ v)[:, None])[0]
print(f"{'arm':26s} {'persona share of orig':>21s} {'cos w/ orig':>11s} {'flipped':>7s} {'HB AUC':>7s} {'xfer':>6s} {'orig HB':>7s} {'orig xfer':>9s}")
meta, out = {}, {}
for n, v in vec.items():
    o = unit(src[n.replace("_personaonly", "")]); a = sgn_auc(v, Xh, yh); fl = a < 0.5
    if fl: v = -v; a = 1 - a
    xf = float(np.mean([fold(sgn_auc(v, Xf, yf)) for Xf, yf in fam.values()]))
    oa = fold(sgn_auc(o, Xh, yh)); ox = float(np.mean([fold(sgn_auc(o, Xf, yf)) for Xf, yf in fam.values()]))
    sh = float(np.sum((Qp.T @ o) ** 2))
    print(f"{n:26s} {sh:21.3f} {v @ o:11.3f} {str(fl):>7s} {a:7.3f} {xf:6.3f} {oa:7.3f} {ox:9.3f}")
    out[n] = torch.tensor(v, dtype=torch.float32)
    meta[n] = {"counterpart": n.replace("_personaonly", ""), "persona_share_of_counterpart": sh, "cos_counterpart": float(v @ o),
               "sign_flipped": bool(fl), "hb_auc_toward_jb_allrows": float(a), "transfer_auc_folded": xf}
torch.save({"vectors": out, "meta": meta, "layer": 16, "note": "persona-only steering arms, norm_share_controls/build_steer_personaonly.py"},
           HERE / "steer_personaonly_layer16.pt")
