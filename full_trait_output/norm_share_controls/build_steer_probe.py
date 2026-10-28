#!/usr/bin/env python3
"""Bundle for steer_probe.py: w_out (dmu with its persona component removed, = dmu_nopersona of
steer_refdmu_layer16.pt), w_in (dmu's persona component), dmu, refusal (-Arditi), persona (published PersonaJB
steer-w), random_full_s0. All unit norm, oriented toward jailbroken. Writes steer_probe_layer16.pt."""
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
a = torch.load(HERE / "steer_refdmu_layer16.pt", weights_only=False)["vectors"]
b = torch.load(HERE / "steer_controls_v2_layer16.pt", weights_only=False)["vectors"]
T = np.load(HERE.parent / "trait_matrix_layer16.npy").astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
Q = np.linalg.qr(T.T)[0]; dmu = a["dmu"].double().numpy(); pin = Q @ (Q.T @ dmu)
unit = lambda v: v / np.linalg.norm(v)
V = {"w_out": a["dmu_nopersona"].double().numpy(), "w_in": unit(pin), "dmu": dmu, "refusal": a["refusal"].double().numpy(),
     "persona": b["persona"].double().numpy(), "random_full_s0": b["random_full_s0"].double().numpy()}
assert abs(V["w_out"] @ unit(dmu - pin) - 1) < 1e-5
torch.save({"vectors": {k: torch.tensor(unit(v), dtype=torch.float32) for k, v in V.items()}, "layer": 16,
            "note": "steer_probe bundle, build_steer_probe.py"}, HERE / "steer_probe_layer16.pt")
print({k: round(float(unit(v) @ unit(dmu)), 3) for k, v in V.items()}, "(cos with dmu)")
