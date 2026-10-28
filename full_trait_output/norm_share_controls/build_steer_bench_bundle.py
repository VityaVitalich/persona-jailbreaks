#!/usr/bin/env python3
"""One bundle for the JBB / WildJB persona-removal runs: the 4 persona-removed arms (steer_nopersona_layer16.pt),
their 3 with-persona originals, and two same-snapshot references, published PersonaJB steer-w ("persona") and an
isotropic random direction ("random_full_s0"), all copied unchanged from steer_controls_v2_layer16.pt.
Writes steer_bench_layer16.pt."""
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
a = torch.load(HERE / "steer_controls_v2_layer16.pt", weights_only=False)
b = torch.load(HERE / "steer_nopersona_layer16.pt", weights_only=False)
keep = ["alpacajb_d2", "sel_pca_1", "sel_acts_1", "persona", "random_full_s0"]
vec = {**{k: a["vectors"][k] for k in keep}, **b["vectors"]}
meta = {**{k: a["meta"][k] for k in keep}, **b["meta"]}
for k, v in vec.items(): assert abs(float(v.norm()) - 1) < 1e-4, k
torch.save({"vectors": vec, "meta": meta, "layer": 16, "note": "JBB/WildJB persona-removal arms, build_steer_bench_bundle.py"},
           HERE / "steer_bench_layer16.pt")
print(sorted(vec))
