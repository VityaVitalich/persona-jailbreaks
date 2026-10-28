#!/usr/bin/env python3
"""
build_ablation_directions.py

Bundle of unit directions for jailbreaking_tools/ablate_fixed_direction.py: the six
directions of fig:attack plus the random controls, stored under the row keys the
figure uses so the results join straight onto results/steering_cell_cis.json.

  persona            <- w_persona__full          (PersonaJB, trait-classifier normal)
  refusal            <- refusal_pos1_neg__full   (-(Arditi refusal direction), L16 pos -1)
  dmu                <- dmu__full                (mean(jailbroken) - mean(refused))
  utilitarian, closure_seeking, edgy            unit rows of trait_matrix_layer16.npy
  random_full_s{0,1,2}, random_persona_s{0,1,2} from control_directions_layer16.pt

Directional ablation is sign-blind (x - (x.v)v is the same for v and -v), so the
sign conventions of the steering bundle do not matter here and the vectors are
copied as stored.

The bundle also records, per direction, the mean layer-16 last-token projection of
the HarmBench complied and refused pools. Zero-ablation moves every prompt to
projection 0, so these two numbers say where that lands relative to the classes:
below both means (persona, refusal, edgy), between them (dmu, utilitarian), or above
both (closure_seeking, for which zeroing the projection is a small push TOWARD the
jailbreak side). They are for reading the results, not used by the runner.

Writes a SEPARATE bundle; steering_directions_layer16.pt is untouched.

Usage:
  uv run python full_trait_tools/build_ablation_directions.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

FIGURE_ROWS = [("persona", "w_persona__full"),
               ("refusal", "refusal_pos1_neg__full"),
               ("dmu", "dmu__full")]


def unit(v) -> np.ndarray:
    v = np.ascontiguousarray(np.asarray(v, dtype=np.float64).ravel())
    return v / np.sqrt(np.dot(v, v))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steering_bundle",
                   default="full_trait_output/steering_directions/steering_directions_layer16.pt")
    p.add_argument("--control_bundle",
                   default="full_trait_output/steering_directions/control_directions_layer16.pt")
    p.add_argument("--trait_matrix_path", default="full_trait_output/trait_matrix_layer16.npy")
    p.add_argument("--trait_names_path", default="full_trait_output/trait_names_layer16.json")
    p.add_argument("--classified_path",
                   default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    p.add_argument("--activations_path",
                   default="full_trait_output/harmbench_activations/activations.pt")
    p.add_argument("--traits", default="utilitarian,closure_seeking,edgy")
    p.add_argument("--controls",
                   default="random_full_s0,random_full_s1,random_full_s2,"
                           "random_persona_s0,random_persona_s1,random_persona_s2")
    p.add_argument("--output",
                   default="full_trait_output/steering_directions/ablation_directions_layer16.pt")
    args = p.parse_args()

    sb = torch.load(args.steering_bundle, map_location="cpu", weights_only=False)
    cb = torch.load(args.control_bundle, map_location="cpu", weights_only=False)
    V = np.load(args.trait_matrix_path).astype(np.float64)
    names = json.load(open(args.trait_names_path))

    vectors, meta = {}, {}
    for key, src in FIGURE_ROWS:
        vectors[key] = unit(sb["vectors"][src].numpy())
        meta[key] = {"source": f"{args.steering_bundle}:{src}",
                     **{k: v for k, v in sb.get("meta", {}).get(src, {}).items()}}
    for t in [t.strip() for t in args.traits.split(",") if t.strip()]:
        i = names.index(t)
        vectors[t] = unit(V[i])
        meta[t] = {"source": f"{args.trait_matrix_path}[{i}]",
                   "description": f"unit trait direction '{t}', layer 16"}
    for c in [c.strip() for c in args.controls.split(",") if c.strip()]:
        src = f"{c}__full"
        vectors[c] = unit(cb["vectors"][src].numpy())
        meta[c] = {"source": f"{args.control_bundle}:{src}",
                   **{k: v for k, v in cb.get("meta", {}).get(src, {}).items()}}

    # where projection 0 lands relative to the two HarmBench classes at L16 / last token
    rows = [json.loads(l) for l in open(args.classified_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("attack_type") == "human_jailbreak"
            and r.get("jailbroken") is not None]
    acts = torch.load(args.activations_path, map_location="cpu", weights_only=False)
    X = np.stack([acts[r["pair_id"]]["16"].double().numpy() for r in rows if r["pair_id"] in acts])
    y = np.array([int(r["jailbroken"]) for r in rows if r["pair_id"] in acts])
    print(f"HarmBench L16 last-token activations {X.shape}: complied {int(y.sum())}, "
          f"refused {int((1 - y).sum())}\n")

    keys = list(vectors)
    print(f"{'direction':20s} {'|v|':>6s} {'proj complied':>14s} {'proj refused':>13s}  zero lands")
    for k in keys:
        pr = X @ vectors[k]
        mc, mr = float(pr[y == 1].mean()), float(pr[y == 0].mean())
        where = ("below both" if 0 < min(mc, mr) else
                 "above both" if 0 > max(mc, mr) else "between")
        meta[k]["hb_l16_proj_mean_complied"] = mc
        meta[k]["hb_l16_proj_mean_refused"] = mr
        meta[k]["zero_projection_lands"] = where
        print(f"{k:20s} {np.linalg.norm(vectors[k]):6.3f} {mc:14.3f} {mr:13.3f}  {where}")

    fig = [k for k, _ in FIGURE_ROWS] + [t.strip() for t in args.traits.split(",") if t.strip()]
    print("\ncosine between the figure directions")
    print(" " * 18 + "".join(f"{k[:9]:>10s}" for k in fig))
    for a in fig:
        print(f"{a:18s}" + "".join(f"{float(np.dot(vectors[a], vectors[b])):10.3f}" for b in fig))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": {k: torch.tensor(v, dtype=torch.float32) for k, v in vectors.items()},
                "meta": meta,
                "layer": 16,
                "note": "unit directions for directional ablation (sign-blind); "
                        "layer 16 is where they were estimated, the ablation is applied "
                        "at every layer"},
               out)
    print(f"\nwrote {len(vectors)} vectors -> {out}")


if __name__ == "__main__":
    main()
