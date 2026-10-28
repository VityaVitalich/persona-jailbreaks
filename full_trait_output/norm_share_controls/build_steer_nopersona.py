#!/usr/bin/env python3
"""
Persona-removed steering arms: do the Alpaca-derived steering directions lose their effect when the persona
component is removed? Each arm has a with-persona counterpart already run on the official snapshot
(steer_controls_v2_layer16.pt, outputs_controls_v2/alpaca{1,2}):
  alpacajb_d2_nopersona  PersonaJB detector (StandardScaler + LR C=10, all 3339 labelled HB rows) refit on the
                         229 Alpaca activations of acts_d2 with their persona-span component removed; normal mapped
                         back. Lies entirely outside the persona span.            counterpart: alpacajb_d2
  alpacajb_d2_nonr       same, but the component in a 229-dim No Robots activation span (nr_d4) removed instead
                         (removes more of each vector than the persona span does): specificity control
  sel_pca_1_nopersona    sel_pca_1 (top Alpaca PC by |AUC-0.5|) minus its persona-span component  counterpart: sel_pca_1
  sel_acts_1_nopersona   sel_acts_1 (top Alpaca prompt activation) minus its persona-span component counterpart: sel_acts_1
All unit norm. Writes steer_nopersona_layer16.pt (steer_fixed_direction.py format) and prints HB / transfer AUC.
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
T = unit_rows(np.load(cfg["T"]).astype(np.float64))
cb, eb = np.load(HERE / "control_bases_llama.npz"), np.load(HERE / "extra_bases_llama.npz")
old = torch.load(HERE / "steer_controls_v2_layer16.pt", weights_only=False)
X, y, _ = S.build_matrix(S.load_jsonl(cfg["human"][0]), torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
X = X.astype(np.float64)
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
fam = {f: S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)[:2] for f, (rp, ap) in cfg["fams"].items()}
Qp = np.linalg.qr(T.T)[0]; Qn = np.linalg.qr(eb["nr_d4"].astype(np.float64).T)[0]
drop = lambda M, Q: M - (M @ Q) @ Q.T

def det(V):
    m = S.model(0).fit(X @ V.T, y); return unit(V.T @ (m[-1].coef_[0] / m[0].scale_))
A = unit_rows(cb["acts_d2"].astype(np.float64))
vec = {"alpacajb_d2_nopersona": det(unit_rows(drop(A, Qp))), "alpacajb_d2_nonr": det(unit_rows(drop(A, Qn))),
       "sel_pca_1_nopersona": unit(drop(old["vectors"]["sel_pca_1"].double().numpy()[None], Qp)[0]),
       "sel_acts_1_nopersona": unit(drop(old["vectors"]["sel_acts_1"].double().numpy()[None], Qp)[0])}
pairs = {"alpacajb_d2_nopersona": "alpacajb_d2", "alpacajb_d2_nonr": "alpacajb_d2",
         "sel_pca_1_nopersona": "sel_pca_1", "sel_acts_1_nopersona": "sel_acts_1"}
P = old["vectors"]["persona"].double().numpy(); R = old["vectors"]["refusal"].double().numpy()
sgn_auc = lambda v, Xm, ym: rank_auc(np.asarray(ym), (np.asarray(Xm, np.float64) @ v)[:, None])[0]
print(f"{'arm':24s} {'persona share':>13s} {'cos w/ orig':>11s} {'HB AUC':>7s} {'xfer AUC':>8s} {'cos PJB':>8s} {'cos -ref':>8s}")
meta = {}
for n, v in list(vec.items()) + [(o, old["vectors"][o].double().numpy()) for o in ("alpacajb_d2", "sel_pca_1", "sel_acts_1")]:
    v = unit(v); o = old["vectors"][pairs.get(n, n)].double().numpy()
    sh = float(np.sum((Qp.T @ v) ** 2)); hbauc = sgn_auc(v, Xh, yh)
    xf = float(np.mean([fold(sgn_auc(v, Xf, yf)) for Xf, yf in fam.values()]))
    print(f"{n:24s} {sh:13.3f} {v @ o:11.3f} {fold(hbauc):7.3f} {xf:8.3f} {v @ P:+8.3f} {-(v @ R):+8.3f}")
    if n in pairs:
        assert hbauc > 0.5, n  # oriented toward jailbroken
        meta[n] = {"counterpart": pairs[n], "persona_share": sh, "hb_auc_toward_jb": float(hbauc), "transfer_auc_folded": xf,
                   "cos_persona": float(v @ P), "cos_refusal_neg": float(-(v @ R))}
torch.save({"vectors": {n: torch.tensor(unit(v), dtype=torch.float32) for n, v in vec.items()}, "meta": meta, "layer": 16,
            "note": "persona-removed steering arms, norm_share_controls/build_steer_nopersona.py"}, HERE / "steer_nopersona_layer16.pt")
