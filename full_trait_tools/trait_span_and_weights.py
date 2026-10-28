#!/usr/bin/env python3
"""
trait_span_and_weights.py

Produces the two remaining mechanistic numbers for a model, in one pass:

  1. Sufficiency (tab:subspace-projection): for each fixed scoring direction
     (w_raw, the class-mean pattern dmu, and random unit directions), the share of
     its squared norm that lies inside the trait span, and the held-out detection
     AUC of the direction, of its in-span part P d, and of its residual (I-P) d.

  2. Signal-carrying traits (tab:trait-jailbreak): the trait projections with the
     largest logistic-regression weights, split by the side they push toward, and
     the traits whose directions are most aligned with w_raw.

P is the orthogonal projector onto the row space of the trait matrix, obtained by
QR. The split is the same behaviour- and template-disjoint pool split used
elsewhere, at a fixed seed, so the test set matches the other single-split tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_all_traits_sweep_v2 import (
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    load_trait_matrix,
    project_all_traits,
    safe_auc,
    split_by_pool,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--human_classified_path",
                   default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    p.add_argument("--human_activations_path",
                   default="full_trait_output/harmbench_activations/activations.pt")
    p.add_argument("--w_raw_path",
                   default="full_trait_output/harmbench_logreg/hyperplane_normal_layer16.pt")
    p.add_argument("--trait_vectors_dir", default=None)
    p.add_argument("--trait_row_index", type=int, default=None)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_random", type=int, default=50)
    p.add_argument("--n_top", type=int, default=10)
    p.add_argument("--augment_classified_path", default=None,
                   help="Fold these rows into the classifier training pool only "
                        "(directions and the held-out split stay pure HarmBench). "
                        "Gemma needs WildJailbreak here: HarmBench alone leaves ~44 "
                        "jailbroken prompts in the train pool.")
    p.add_argument("--augment_activations_path", default=None)
    p.add_argument("--output_dir", default="full_trait_output/trait_span_and_weights")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_jsonl(Path(args.human_classified_path))
            if r.get("attack_type") == "human_jailbreak"]
    acts = torch.load(Path(args.human_activations_path), map_location="cpu", weights_only=False)
    x_raw, y, valid = build_activation_matrix(rows, acts, args.layer)

    V, names = load_trait_matrix(
        args.layer,
        vectors_dir=Path(args.trait_vectors_dir) if args.trait_vectors_dir else None,
        cache_dir=out_dir,
        row_index=args.trait_row_index,
    )
    print(f"activations {x_raw.shape}, traits {V.shape}, jb rate {y.mean():.3f}", flush=True)

    tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(valid, args.train_frac, args.seed)
    tr, te = split_by_pool(valid, tr_beh, tr_tpl, te_beh, te_tpl)
    print(f"train {len(tr)}, test {len(te)} (test jb {int(y[te].sum())})", flush=True)

    # Orthonormal basis for the trait span, then the projector applied via Q.
    Q, _ = np.linalg.qr(V.T)                      # [hidden, rank]
    def proj(d):
        return Q @ (Q.T @ d)

    w_raw = torch.load(Path(args.w_raw_path), map_location="cpu",
                       weights_only=False)
    if isinstance(w_raw, dict):
        w_raw = w_raw.get("vector", w_raw.get("w", w_raw.get("normal")))
    w_raw = np.asarray(w_raw, dtype=np.float64).ravel()
    w_raw /= np.linalg.norm(w_raw)

    dmu = x_raw[tr][y[tr] == 1].mean(0) - x_raw[tr][y[tr] == 0].mean(0)
    dmu = dmu.astype(np.float64) / np.linalg.norm(dmu)

    def auc_of(d):
        return safe_auc(y[te], x_raw[te] @ d)

    results = {}
    for name, d in (("w_raw", w_raw), ("dmu", dmu)):
        pd_ = proj(d)
        rd = d - pd_
        share = float(np.dot(pd_, pd_) / np.dot(d, d))
        results[name] = {
            "share_in_span": share,
            "auc_full": auc_of(d),
            "auc_proj": auc_of(pd_ / (np.linalg.norm(pd_) + 1e-12)),
            "auc_resid": auc_of(rd / (np.linalg.norm(rd) + 1e-12)),
        }
        print(f"{name:6s} share_in_span={share:.4f} auc={results[name]['auc_full']:.4f} "
              f"proj={results[name]['auc_proj']:.4f} resid={results[name]['auc_resid']:.4f}",
              flush=True)

    rng = np.random.RandomState(args.seed)
    r_aucs, r_shares = [], []
    for _ in range(args.n_random):
        d = rng.randn(x_raw.shape[1]).astype(np.float64)
        d /= np.linalg.norm(d)
        r_aucs.append(auc_of(d))
        r_shares.append(float(np.dot(proj(d), proj(d)) / np.dot(d, d)))
    results["random"] = {
        "auc_mean": float(np.mean(r_aucs)), "auc_std": float(np.std(r_aucs)),
        "share_in_span_mean": float(np.mean(r_shares)),
    }
    print(f"random auc={results['random']['auc_mean']:.4f} "
          f"+/-{results['random']['auc_std']:.4f} "
          f"share={results['random']['share_in_span_mean']:.4f}", flush=True)

    # ── Signal-carrying traits ────────────────────────────────────────────────
    xt = project_all_traits(x_raw, V)
    x_fit, y_fit = xt[tr], y[tr]
    if args.augment_classified_path and args.augment_activations_path:
        a_rows = load_jsonl(Path(args.augment_classified_path))
        a_acts = torch.load(Path(args.augment_activations_path), map_location="cpu",
                            weights_only=False)
        ax, ay, _ = build_activation_matrix(a_rows, a_acts, args.layer)
        a_xt = project_all_traits(ax, V)
        x_fit = np.vstack([x_fit, a_xt])
        y_fit = np.concatenate([y_fit, ay])
        print(f"augment(train only): +{len(ax)} rows, +{int(ay.sum())} jb "
              f"-> {int(y_fit.sum())} training positives", flush=True)
    sc = StandardScaler().fit(x_fit)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(sc.transform(x_fit), y_fit)
    coef = clf.coef_.ravel()
    order = np.argsort(coef)
    safe_side = [(names[i], float(coef[i])) for i in order[:args.n_top]]
    jb_side = [(names[i], float(coef[i])) for i in order[::-1][:args.n_top]]

    cos_w = V @ w_raw                              # traits are unit-norm rows
    aligned = [(names[i], float(cos_w[i]))
               for i in np.argsort(-np.abs(cos_w))[:args.n_top]]

    print("\nsafe side   :", ", ".join(f"{n} ({c:+.2f})" for n, c in safe_side))
    print("jailbroken  :", ", ".join(f"{n} ({c:+.2f})" for n, c in jb_side))
    print("aligned w   :", ", ".join(f"{n} ({c:+.2f})" for n, c in aligned))

    out = {
        "layer": args.layer, "n_traits": len(names),
        "n_train": len(tr), "n_test": len(te),
        "trait_span_rank": int(Q.shape[1]),
        "directions": results,
        "largest_weight_safe_side": safe_side,
        "largest_weight_jailbroken_side": jb_side,
        "most_aligned_with_w_raw": aligned,
        "test_auc_trait_probe": safe_auc(y[te], clf.decision_function(sc.transform(xt[te]))),
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
