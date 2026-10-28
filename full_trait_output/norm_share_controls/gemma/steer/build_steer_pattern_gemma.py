#!/usr/bin/env python3
"""Gemma steering arm: the activation pattern of the PersonaJB detector (Haufe et al. 2014), kept in persona space.
  w        PersonaJB detector normal of the Gemma bundle (w_persona__full; scaler + LR C=1 on the 240 trait
           projections, HB human_jailbreak + WJB, mapped to hidden space)
  a        Cov w, Cov = covariance of the layer-30 activations of the same HB + WJB prompts (= cov(h, w.h))
  arm      persona_pattern = a projected onto the span of the 240 trait vectors, unit norm, oriented toward
           jailbroken by the mean projection gap on HB + WJB (the bundle's rule)
Prints cosines with w, the unprojected pattern, dmu and refusal; share of a in the trait span; spread along the arm
(x random direction) and folded HB / transfer AUC. Writes steer_pattern_gemma_layer30.pt."""
import sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
G = Path(__file__).resolve().parent.parent; O = G.parent.parent; sys.path.insert(0, str(O / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402
cfg = S.CFG["gemma"]; unit = lambda v: v / np.linalg.norm(v); fa = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True); Qp = np.linalg.qr(T.T)[0]
B = torch.load(O / "gemma_steering_inputs/steering_directions_gemma_layer30.pt", weights_only=False)["vectors"]
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 30)
Xa, ya, _ = S.build_matrix(S.load_jsonl(cfg["aug"][0]), torch.load(cfg["aug"][1], map_location="cpu", weights_only=False), 30)
XF = np.vstack([Xh, Xa]).astype(np.float64); yF = np.r_[yh, ya]; Xc = XF - XF.mean(0)
fam = {f: S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 30)[:2] for f, (rp, ap) in cfg["fams"].items()}
w = unit(B["w_persona__full"].double().numpy())
a = unit(Xc.T @ (Xc @ w) / (len(Xc) - 1))
p = unit(Qp @ (Qp.T @ a)); gap = lambda v: float((XF[yF == 1] @ v).mean() - (XF[yF == 0] @ v).mean())
if gap(p) < 0: p = -p; print("flipped")
R = np.random.default_rng(0).standard_normal((200, XF.shape[1])); R /= np.linalg.norm(R, axis=1, keepdims=True); sd_r = np.median((Xc @ R.T).std(0))
for n, v in (("w (current PersonaJB)", w), ("pattern a (full space)", a), ("persona_pattern (arm)", p),
             ("dmu", unit(B["dmu__full"].double().numpy())), ("refusal", unit(B["refusal_pos1_neg__full"].double().numpy()))):
    print(f"{n:24s} cos arm {v @ p:+.3f} | cos w {v @ w:+.3f} | share in trait span {np.sum((Qp.T @ v) ** 2):.3f} | spread x random {(Xc @ v).std() / sd_r:5.1f} | "
          f"push at a=.2 {0.2 * 258.088 / (Xc @ v).std():5.1f} sigma | HB AUC {fa(yh, Xh @ v):.3f} | transfer {np.mean([fa(y, X @ v) for X, y in fam.values()]):.3f}")
torch.save({"vectors": {"persona_pattern": torch.tensor(p, dtype=torch.float32)}, "layer": 30,
            "meta": {"persona_pattern": {"description": "trait-span projection of Cov w, w = w_persona__full, Cov over HB+WJB layer-30 activations", "mean_gap_hbwjb": gap(p)}},
            "note": "gemma/steer/build_steer_pattern_gemma.py"}, Path(__file__).resolve().parent / "steer_pattern_gemma_layer30.pt")
