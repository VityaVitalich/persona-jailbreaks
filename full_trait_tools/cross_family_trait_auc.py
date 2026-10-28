#!/usr/bin/env python3
"""
cross_family_trait_auc.py

Per-attack-family single-trait detection AUC, to ask whether the same persona
directions carry the jailbreak signal across benchmarks (interp L115: "why transfer
happens"). For each family we project the layer-16 activation onto one trait vector at
a time and score the 1-D feature (safe_auc, orientation-invariant), exactly as the
single-trait-AUC analysis does on HarmBench. We then compare the top-k trait sets
across families (overlap, rank correlation) and identify traits that stay predictive
everywhere.

Emits results.json and, if matplotlib is available, a trait x family AUC heatmap.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from run_all_traits_sweep_v2 import (
    build_activation_matrix,
    load_jsonl,
    load_trait_matrix,
    safe_auc,
)

LAYER = 16
PD = "full_trait_output/paper_data/activations"

# (name, labels_jsonl, activations_pt, attack_type_filter or None)
FAMILIES = [
    ("HarmBench", f"{PD}/harmbench/classified_responses.jsonl", f"{PD}/harmbench/activations.pt", "human_jailbreak"),
    ("PAIR",      f"{PD}/pair/responses.jsonl",                 f"{PD}/pair/activations.pt",       None),
    ("PAP",       f"{PD}/pap/responses.jsonl",                  f"{PD}/pap/activations.pt",        None),
    ("GPTFuzz",   f"{PD}/gptfuzz/responses.jsonl",              f"{PD}/gptfuzz/activations.pt",    None),
    ("PEZ",       f"{PD}/pez/responses.jsonl",                  f"{PD}/pez/activations.pt",        None),
    ("GCG",       f"{PD}/gcg/responses.jsonl",                  f"{PD}/gcg/activations.pt",        None),
    ("WildJB",    f"{PD}/wildjailbreak/classified_responses.jsonl", f"{PD}/wildjailbreak/activations.pt", None),
]


# Gemma keeps its own copies under full_trait_output rather than the paper_data
# symlinks, and has no GCG family until that generation run lands.
FAMILIES_GEMMA = [
    ("HarmBench", "full_trait_output/harmbench_activations_gemma/classified_responses.jsonl",
     "full_trait_output/harmbench_activations_gemma/activations.pt", "human_jailbreak"),
    ("PAIR",      "full_trait_output/pair_activations_gemma_hb/classified_responses.jsonl",
     "full_trait_output/pair_activations_gemma_hb/activations.pt", None),
    ("PAP",       "full_trait_output/pap_activations_gemma_hb/classified_responses.jsonl",
     "full_trait_output/pap_activations_gemma_hb/activations.pt", None),
    ("GPTFuzz",   "full_trait_output/gptfuzz_activations_gemma_hb/classified_responses.jsonl",
     "full_trait_output/gptfuzz_activations_gemma_hb/activations.pt", None),
    ("PEZ",       "full_trait_output/pez_activations_gemma_hb/classified_responses.jsonl",
     "full_trait_output/pez_activations_gemma_hb/activations.pt", None),
    ("GCG",       "full_trait_output/gcg_activations_gemma_hb/classified_responses.jsonl",
     "full_trait_output/gcg_activations_gemma_hb/activations.pt", None),
    ("WildJB",    "full_trait_output/wildjailbreak_activations_gemma/classified_responses.jsonl",
     "full_trait_output/wildjailbreak_activations_gemma/activations.pt", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--gemma", action="store_true",
                    help="Use the Gemma family paths and its layer-30 trait vectors.")
    ap.add_argument("--trait_vectors_dir", default=None,
                    help="Build the trait matrix from these .pt files instead of the "
                         "default Llama cache.")
    ap.add_argument("--trait_row_index", type=int, default=None,
                    help="Row of each saved vector to use — Gemma layer30_only needs 0.")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--output_dir", default="full_trait_output/cross_family_traits")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    family_specs = FAMILIES_GEMMA if args.gemma else FAMILIES
    trait_matrix, trait_names = load_trait_matrix(
        args.layer,
        vectors_dir=Path(args.trait_vectors_dir) if args.trait_vectors_dir else None,
        cache_dir=Path(args.output_dir),
        row_index=args.trait_row_index)  # [229, 4096], unit-norm rows
    n_traits = trait_matrix.shape[0]
    print(f"Trait matrix: {trait_matrix.shape}", flush=True)

    fam_auc = {}   # family -> np.array[n_traits] (nan where undefined)
    fam_meta = {}
    for name, lbl_path, act_path, atype in family_specs:
        if not Path(lbl_path).exists() or not Path(act_path).exists():
            print(f"  [skip] {name}: missing files", flush=True)
            continue
        rows = load_jsonl(Path(lbl_path))
        if atype is not None:
            rows = [r for r in rows if r.get("attack_type") == atype]
        acts = torch.load(Path(act_path), map_location="cpu", weights_only=False)
        x, y, valid = build_activation_matrix(rows, acts, args.layer)
        del acts
        if len(y) == 0 or len(np.unique(y)) < 2:
            print(f"  [skip] {name}: n={len(y)}, single-class", flush=True)
            continue
        proj = x @ trait_matrix.T  # [n, n_traits]
        aucs = np.array([safe_auc(y, proj[:, t]) for t in range(n_traits)])
        fam_auc[name] = aucs
        fam_meta[name] = {"n_pairs": int(len(y)), "jb_rate": float(y.mean())}
        order = np.argsort(-aucs)
        top = [(trait_names[t], round(float(aucs[t]), 4)) for t in order[:args.top_k]]
        print(f"  {name:10s} n={len(y):5d} jb={y.mean():.2f}  top: "
              + ", ".join(f"{nm}({a:.2f})" for nm, a in top[:5]), flush=True)
        fam_meta[name]["top_k"] = top

    families = list(fam_auc.keys())

    # --- top-k overlap (Jaccard) + rank correlation between families ---
    topk_sets = {f: {trait_names[t] for t in np.argsort(-fam_auc[f])[:args.top_k]} for f in families}
    jaccard = {}
    spearman = {}
    for a, b in itertools.combinations(families, 2):
        inter = len(topk_sets[a] & topk_sets[b])
        union = len(topk_sets[a] | topk_sets[b])
        jaccard[f"{a}|{b}"] = {"intersection": inter, "jaccard": round(inter / union, 4)}
        rho = spearmanr(fam_auc[a], fam_auc[b]).correlation
        spearman[f"{a}|{b}"] = round(float(rho), 4)

    global_inter = set.intersection(*topk_sets.values()) if families else set()

    # --- stability: mean/std/min AUC across families, and top-k appearance count ---
    A = np.vstack([fam_auc[f] for f in families])  # [n_fam, n_traits]
    appear = {trait_names[t]: sum(trait_names[t] in topk_sets[f] for f in families) for t in range(n_traits)}
    stability = []
    for t in range(n_traits):
        stability.append({
            "trait": trait_names[t],
            "mean_auc": round(float(A[:, t].mean()), 4),
            "min_auc": round(float(A[:, t].min()), 4),
            "std_auc": round(float(A[:, t].std()), 4),
            "topk_appearances": appear[trait_names[t]],
        })
    stability.sort(key=lambda d: (-d["topk_appearances"], -d["mean_auc"]))

    out = {
        "method": "cross_family_single_trait_auc",
        "layer": args.layer,
        "top_k": args.top_k,
        "families": families,
        "family_meta": fam_meta,
        "topk_global_intersection": sorted(global_inter),
        "pairwise_topk_overlap": jaccard,
        "pairwise_spearman_rank_corr": spearman,
        "stability_ranking": stability,
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nGlobal top-{args.top_k} intersection ({len(global_inter)}): {sorted(global_inter)}", flush=True)
    print("Most stable traits (top-k appearances, mean AUC):", flush=True)
    for s in stability[:12]:
        print(f"  {s['trait']:16s} appears in {s['topk_appearances']}/{len(families)}  "
              f"mean={s['mean_auc']:.3f} min={s['min_auc']:.3f}", flush=True)

    # --- heatmap: union of each family's top-k, ordered by mean AUC ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        union = sorted(set().union(*topk_sets.values()),
                       key=lambda nm: -A[:, trait_names.index(nm)].mean())
        name_to_idx = {nm: trait_names.index(nm) for nm in union}
        M = np.array([[fam_auc[f][name_to_idx[nm]] for f in families] for nm in union])

        fig, ax = plt.subplots(figsize=(1.1 * len(families) + 2.5, 0.34 * len(union) + 1.5))
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=max(0.75, float(M.max())))
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=30, ha="right")
        ax.set_yticks(range(len(union)))
        ax.set_yticklabels(union, fontsize=8)
        for i in range(len(union)):
            for j in range(len(families)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.65 else "black", fontsize=6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("single-trait AUC")
        ax.set_title(f"Single-trait detection AUC by attack family (layer {args.layer})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "cross_family_trait_auc_heatmap.png", dpi=200)
        print(f"\nSaved heatmap to {out_dir / 'cross_family_trait_auc_heatmap.png'}", flush=True)
    except Exception as e:
        print(f"\n[heatmap skipped] {e}", flush=True)


if __name__ == "__main__":
    main()
