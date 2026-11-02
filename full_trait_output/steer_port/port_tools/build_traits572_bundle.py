# Build inputs/gemma/steer_traits572_gemma_layer30.pt: unit trait directions for the norm-572 comparison run.
# steer_<t>: top-10 all-trait-sweep traits, positive pole (as in the sweep).
# pred_<t>: top-10 predictive traits, signed toward the side that predicts jailbreak (as on the dashboard).
import json, sys, numpy as np, torch
P = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/steer_port"
M = np.load(f"{P}/inputs/gemma/alltraits/trait_matrix_layer30.npy"); names = json.load(open(f"{P}/inputs/gemma/alltraits/trait_names_layer30.json"))
names = names if isinstance(names, list) else list(names); idx = {n: i for i, n in enumerate(names)}
spec = json.load(open(f"{P}/inputs/gemma/traits572_spec.json"))
base = torch.load(f"{P}/inputs/gemma/steer_norm572_gemma_layer30.pt", map_location="cpu", weights_only=False)
vec, meta = {"random_full_s0": base["vectors"]["random_full_s0"]}, {"random_full_s0": base.get("meta", {}).get("random_full_s0", {})}
for grp, items in spec.items():
    for t, sgn in items:
        v = sgn * M[idx[t]].astype(np.float64); v /= np.linalg.norm(v)
        vec[f"{grp}_{t}"] = torch.tensor(v, dtype=torch.float32)
        meta[f"{grp}_{t}"] = {"description": f"trait {t}, sign {sgn:+d} ({'sweep top-10, positive pole' if grp == 'steer' else 'top-10 predictive, toward jailbreak side'})"}
out = f"{P}/inputs/gemma/steer_traits572_gemma_layer30.pt"
import os; assert not os.path.exists(out), "exists"
torch.save({"vectors": vec, "meta": meta, "layer": base.get("layer", 30)}, out)
print("saved", len(vec), sorted(vec), "layer", base.get("layer"), "dim", vec["random_full_s0"].numel())
