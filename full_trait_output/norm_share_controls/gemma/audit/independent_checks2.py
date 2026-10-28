#!/usr/bin/env python3
"""Audit 2: (a) spread of transfer AUC across dictionary draws (per-draw values from the ported run);
(b) refusal direction with / without its persona component, recomputed from the raw harmful/harmless means
(gemma_refusal_dir, hidden_states[31], last position), sklearn AUC, HB all rows and test pools."""
import json, sys, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
G = Path(__file__).resolve().parent.parent; O = G.parent.parent
r = json.load(open(G / "remove_persona_all_controls.json"))
print("(a) per-draw transfer AUC (whole -> persona removed)")
for n, outs in r.items(): print(f"   {n:20s} " + "  ".join(f"{o['whole'][1]:.3f}->{o['persona'][2]:.3f}" for o in outs))
exec(open(Path(__file__).resolve().parent / "independent_removal.py").read().split("rp = [json.loads")[0])
mh = torch.load(O / "gemma_refusal_dir/out/means_harmful.pt", weights_only=False); ml = torch.load(O / "gemma_refusal_dir/out/means_harmless.pt", weights_only=False)
ref = (mh[31, -1] - ml[31, -1]).double().numpy(); ref /= np.linalg.norm(ref)
b = torch.load(O / "gemma_steering_inputs/steering_directions_gemma_layer30.pt", weights_only=False)["vectors"]["refusal_pos1_neg__full"].double().numpy()
print("(b) cos(raw-means refusal, bundle refusal) =", round(float(ref @ b / np.linalg.norm(b)), 4))
Qp = np.linalg.qr(T.T)[0]; pin = Qp @ (Qp.T @ ref)
fa = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
for n, v in (("refusal", ref), ("refusal persona component", pin / np.linalg.norm(pin)), ("refusal persona removed", (ref - pin) / np.linalg.norm(ref - pin))):
    print(f"   {n:28s} share {float(np.sum((Qp.T @ v) ** 2)):.3f} | HB all rows {fa(yh, Xh @ v):.3f} | " + " ".join(f"{f} {fa(y, X @ v):.3f}" for f, (X, y) in fam.items()) + f" | mean5 {np.mean([fa(y, X @ v) for X, y in fam.values()]):.3f}")
