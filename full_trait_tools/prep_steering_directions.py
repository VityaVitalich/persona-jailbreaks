#!/usr/bin/env python3
"""
prep_steering_directions.py

Builds the fixed directions that jailbreaking_tools/steer_fixed_direction.py
steers with, and audits them against the numbers already in the paper.

Directions:
  refusal_pos1_neg  -(Arditi refusal direction), layer 16, position -1
  refusal_pos5_neg  -(Arditi refusal direction), layer 16, position -5
  dmu                mean(jailbroken) - mean(refused) over all HarmBench
                     human_jailbreak pairs -- the unsupervised jailbreak
                     direction of fig:direction-story
w_persona and w_raw are written too, as same-run anchors against the published
tab:attack rows.

Every vector is unit-norm, so the steering hook (alpha * RESIDUAL_NORM * v)
applies the same magnitude to every arm and the arms differ only in direction.

The audit reproduces the geometry the paper reports: in-span share 0.407 / 0.209
for the refusal direction at position -1 / -5 (tab:refusal-direction) and 0.482
for dmu (the 48.2% of fig:direction-story). The refusal direction is Arditi's,
computed on separate data, so it is a check that the projector and the vectors
line up; dmu depends on the pool it is estimated over, which is why the
split-restricted variant is printed next to the full-pool one.

Usage:
  uv run python full_trait_tools/prep_steering_directions.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_traits_sweep_v2 import (  # noqa: E402
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    project_all_traits,
    safe_auc,
    split_by_pool,
)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / np.linalg.norm(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classified_path",
                   default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    p.add_argument("--activations_path",
                   default="full_trait_output/harmbench_activations/activations.pt")
    p.add_argument("--trait_matrix_path",
                   default="full_trait_output/trait_matrix_layer16.npy")
    p.add_argument("--w_raw_path",
                   default="full_trait_output/harmbench_logreg/hyperplane_normal_layer16.pt")
    p.add_argument("--refusal_dir",
                   default=str(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct"),
                   help="Arditi recreation: direction_layer16{,_lasttoken}.pt")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="full_trait_output/steering_directions")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_jsonl(Path(args.classified_path))
            if r.get("attack_type") == "human_jailbreak"]
    acts = torch.load(Path(args.activations_path), map_location="cpu",
                      weights_only=False)
    x_raw, y, valid = build_activation_matrix(rows, acts, args.layer)
    tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(valid, args.train_frac, args.seed)
    tr, te = split_by_pool(valid, tr_beh, tr_tpl, te_beh, te_tpl)
    print(f"activations {x_raw.shape}  jb_rate={y.mean():.3f}  "
          f"train={len(tr)} test={len(te)} (test jb {int(y[te].sum())})", flush=True)

    V = np.asarray(np.load(args.trait_matrix_path), dtype=np.float64)
    Q, _ = np.linalg.qr(V.T)
    print(f"trait matrix {V.shape}  span rank {Q.shape[1]}", flush=True)

    def proj(d):
        return Q @ (Q.T @ d)

    def auc(d):
        return safe_auc(y[te], x_raw[te] @ d)

    w_raw = torch.load(Path(args.w_raw_path), map_location="cpu", weights_only=False)
    if isinstance(w_raw, dict):
        w_raw = w_raw.get("vector", w_raw.get("w", w_raw.get("normal")))
    w_raw = unit(w_raw)

    # The steer-w (persona) direction of tab:attack, rebuilt with the recipe of
    # trait_steering_attack_jbb.fit_classifier: all human_jailbreak pairs (the
    # published steering run is in-sample), float32 projections, StandardScaler,
    # LogisticRegression(max_iter=2000, C=1.0, random_state=42). The 5-fold CV
    # AUC printed below must match the 0.968131 that the published JBB persona_w
    # summary recorded for its classifier, which is what certifies this is the
    # same vector rather than a lookalike.
    xt = project_all_traits(x_raw, V).astype(np.float32)
    sc = StandardScaler().fit(xt)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(sc.transform(xt), y)
    w_persona = unit((clf.coef_.ravel() / sc.scale_) @ V)
    cv = cross_val_score(
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
        xt, y, cv=5, scoring="roc_auc").mean()
    print(f"w_persona classifier: n={len(y)} jb_rate={y.mean():.4f} "
          f"cv_auc={cv:.6f} (published run recorded 0.968131)", flush=True)

    # dmu over the whole pool (the figure's 48.2%); the split-restricted variant
    # is audited below but not steered with, so the direction does not depend on
    # a split the steering run has no other use for.
    dmu = unit(x_raw[y == 1].mean(0) - x_raw[y == 0].mean(0))
    dmu_train = unit(x_raw[tr][y[tr] == 1].mean(0) - x_raw[tr][y[tr] == 0].mean(0))

    rd = Path(args.refusal_dir)
    ref1 = torch.load(rd / "direction_layer16_lasttoken.pt", map_location="cpu",
                      weights_only=False)
    ref5 = torch.load(rd / "direction_layer16.pt", map_location="cpu",
                      weights_only=False)
    print(f"refusal directions from {ref1['model']} "
          f"(layer {ref1['layer']}, positions {ref1['position']} / {ref5['position']})",
          flush=True)

    # Arditi's r points toward refusal, so the jailbreak-pushing direction is -r.
    steerable = [
        ("refusal_pos1_neg", -unit(ref1["direction"].numpy()),
         "-(Arditi refusal direction), layer 16 pos -1"),
        ("refusal_pos5_neg", -unit(ref5["direction"].numpy()),
         "-(Arditi refusal direction), layer 16 pos -5"),
        ("dmu", dmu, "unsupervised jailbreak direction, mean(jb) - mean(refused), all pairs"),
    ]
    anchors = [
        ("w_persona", w_persona, "trait-classifier normal (published steer-w persona)"),
        ("w_raw", w_raw, "raw-activation classifier normal (published steer-w raw)"),
    ]

    print(f"\n{'direction':18s} {'in-span':>8s} {'AUC':>7s} {'AUCproj':>8s} {'AUCres':>7s}",
          flush=True)
    audit = {}
    for name, d, _ in steerable + anchors + [("dmu_train_pool", dmu_train, "")]:
        pd_, rs = proj(d), d - proj(d)
        share = float(pd_ @ pd_ / (d @ d))
        audit[name] = dict(share_in_span=share, auc=auc(d), auc_proj=auc(unit(pd_)),
                           auc_resid=auc(unit(rs)))
        print(f"{name:18s} {share:8.3f} {audit[name]['auc']:7.3f} "
              f"{audit[name]['auc_proj']:8.3f} {audit[name]['auc_resid']:7.3f}", flush=True)

    rng = np.random.RandomState(args.seed)
    shares = []
    for _ in range(50):
        u = unit(rng.randn(x_raw.shape[1]))
        shares.append(float(proj(u) @ proj(u)))
    audit["random_span_share"] = float(np.mean(shares))
    print(f"{'random-229 span':18s} {np.mean(shares):8.3f}", flush=True)

    all_named = steerable + anchors
    keys = [n for n, _, _ in all_named]
    print("\ncosine between directions")
    print(" " * 18 + "".join(f"{k[:9]:>10s}" for k in keys), flush=True)
    cosines = {}
    for n1, d1, _ in all_named:
        row = [float(d1 @ d2) for _, d2, _ in all_named]
        cosines[n1] = dict(zip(keys, row))
        print(f"{n1:18s}" + "".join(f"{v:10.3f}" for v in row), flush=True)

    vectors, meta = {}, {}
    for name, d, desc in steerable + anchors:
        key = f"{name}__full"
        vectors[key] = torch.tensor(d, dtype=torch.float32)
        meta[key] = dict(base=name, component="full", description=desc,
                         share_in_span=audit[name]["share_in_span"],
                         cos_with_w_persona=float(d @ w_persona),
                         cos_with_dmu=float(d @ dmu))

    torch.save({"vectors": vectors, "meta": meta, "layer": args.layer,
                "split_seed": args.seed, "train_frac": args.train_frac},
               out / "steering_directions_layer16.pt")
    (out / "audit.json").write_text(json.dumps(
        {"layer": args.layer, "n_train": len(tr), "n_test": len(te),
         "trait_span_rank": int(Q.shape[1]), "audit": audit,
         "cosines": cosines, "meta": meta}, indent=2))
    print(f"\nwrote {len(vectors)} vectors -> {out / 'steering_directions_layer16.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
