#!/usr/bin/env python3
"""
per_layer_trait_aucs.py

Persist the full per-layer single-trait AUC matrix that
persona_layer_dynamics.py computes internally but only saves for its
selected personas. Pure re-scoring of existing dumps: projects the layer-L
HarmBench activation onto each persona's layer-L trait vector and computes
the orientation-invariant AUC. No extraction, no classifier fitting.

Validates against persona_layer_dynamics/results.json before writing, and
prints the traits whose AUC peaks at the paper's default layer.

Usage:
  python3 full_trait_tools/per_layer_trait_aucs.py
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ACTS = "full_trait_output/all_layers_activations/harmbench_activations.pt"
ROWS = "full_trait_output/all_layers_activations/harmbench_rows.jsonl"
# identical (md5-verified) to paper_data/building_blocks/persona_trait_vectors_llama
VECS = ("full_trait_output/traits40_vectors/pre_generation_last_token/"
        "filter_matched_pairs_ge_50_count_ge_10_total")
REF = "full_trait_output/persona_layer_dynamics/results.json"
OUT = "full_trait_output/persona_layer_dynamics/auc_by_layer_all229.json"
DEFAULT_LAYER = 16


def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    a = float(roc_auc_score(y, s))
    return max(a, 1.0 - a)


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    label = {}
    for r in rows:
        if r.get("attack_type") != "human_jailbreak":
            continue
        pid = r.get("pair_id")
        jb = r.get("jailbroken", r.get("binary_jailbroken"))
        if pid is not None and jb is not None:
            label[str(pid)] = int(bool(jb))

    acts = torch.load(ACTS, map_location="cpu", weights_only=False)
    keys = [k for k in acts if str(k) in label]
    layers = sorted(int(x) for x in acts[keys[0]].keys())
    y = np.array([label[str(k)] for k in keys])
    print(f"n_pairs={len(keys)}, layers={layers[0]}..{layers[-1]}")

    names, vecs = [], []
    for f in sorted(glob.glob(f"{VECS}/*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        names.append(d.get("trait", os.path.basename(f)[:-3]))
        vecs.append(d["vector"].float().numpy())
    V = np.stack(vecs)  # [n_traits, 32, 4096]
    print(f"loaded {len(names)} trait vectors")

    auc = np.full((len(layers), len(names)), np.nan)
    for li, L in enumerate(layers):
        XL = np.stack([acts[k][str(L)].float().numpy() for k in keys])
        P = XL @ V[:, L, :].T
        for t in range(len(names)):
            auc[li, t] = safe_auc(y, P[:, t])
        print(f"  scored L{L}", flush=True)

    # validate against the saved per-persona curves before writing anything
    ref = json.loads(Path(REF).read_text())["results"]
    idx = {nm: names.index(nm) for nm in
           ["egalitarian", "universalist", "progressive", "principled"]}
    worst = max(abs(auc[li, idx[nm]] - ref[nm][li])
                for nm in idx for li in range(len(layers)))
    print(f"max deviation vs results.json: {worst:.5f}")
    assert worst < 5e-4, "recomputed curves do not match results.json"

    payload = {"layers": layers, "names": names,
               "auc_by_layer": [[round(float(v), 4) for v in row] for row in auc]}
    Path(OUT).write_text(json.dumps(payload))
    print(f"saved {OUT}")

    li16 = layers.index(DEFAULT_LAYER)
    peak_at = np.nanargmax(auc, axis=0)
    print(f"\ntraits peaking at layer {DEFAULT_LAYER} (by AUC there):")
    for t in np.argsort(-auc[li16]):
        if layers[peak_at[t]] == DEFAULT_LAYER:
            print(f"  {names[t]:18s} L16={auc[li16, t]:.3f} "
                  f"(overall rank at L16: {int(np.sum(auc[li16] > auc[li16, t])) + 1})")


if __name__ == "__main__":
    main()
