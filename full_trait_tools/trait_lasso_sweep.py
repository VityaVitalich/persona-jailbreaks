#!/usr/bin/env python3
"""
trait_lasso_sweep.py

How many traits does the PersonaJB direction actually need?

The published PersonaJB direction is a DENSE logistic regression over all 229
trait projections (prep_steering_directions.py):

    xt = x @ V.T ; w_persona = unit((clf.coef_ / scaler.scale_) @ V)

with an L2 penalty, so every trait contributes. This script replaces the L2
penalty with L1 and sweeps the regularisation strength, which turns the fit into
a feature-selection problem over traits: at each C only a subset of the 229 rows
of V carries weight, and w lives in the span of that subset. Sweeping C traces
out "the k most predictive traits" for every k, which is the multivariate
generalisation of ranking traits one at a time (persona_individual_auc.py).

Protocol matches run_all_traits_sweep_v2 so the numbers sit next to the paper's:
HarmBench human_jailbreak pairs at layer 16, an outer pool split that is disjoint
in BOTH behaviour and jailbreak template, 50 seeds, transfer families held out
entirely (evaluation only). AUC is the fitted model's oriented ROC AUC, not the
repo's sign-free max(auc, 1-auc): an L1 model carries its own orientation and
flipping it per evaluation would flatter the sparse end of the sweep.

Three selection rules share each budget k, so "sparse is enough" can be separated
from "any k traits are enough". All three hand their chosen columns to the same
unpenalised L2 fit, so the curves differ only in which traits were kept:
  lasso   -- support of the L1 path point with the largest k not exceeding budget
  univar  -- the k traits with the best individual train AUC
  random  -- k traits drawn uniformly (one draw per seed)
The raw L1 path is kept alongside with its own shrunk coefficients; that curve
confounds the choice of traits with the shrinkage the penalty applies to them.

Also recorded per budget: cosine between the k-sparse direction and the dense
PersonaJB direction of the same seed, i.e. how much of the steering vector a
k-trait subspace recovers.

  python full_trait_tools/trait_lasso_sweep.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_traits_sweep_v2 import (  # noqa: E402
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    split_by_pool,
)

# Apple's Accelerate BLAS raises spurious divide/overflow warnings on these
# matmuls; checked against an explicit float64 einsum (max abs diff 1.2e-5).
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.seterr(all="ignore")

REPO = Path(__file__).resolve().parents[1]
KGRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 229]
TRANSFER = [
    ("GCG", "gcg_activations/responses.jsonl"),
    ("PAIR", "pair_activations/responses.jsonl"),
    ("PAP", "pap_activations/responses.jsonl"),
    ("GPTFuzz", "gptfuzz_activations/responses.jsonl"),
    ("PEZ", "pez_activations/responses.jsonl"),
    ("WJB", "wildjailbreak_activations/classified_responses.jsonl"),
]


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def fit_l2(xt_tr, y_tr, cols):
    """Dense L2 logistic regression restricted to `cols`, the paper's recipe."""
    sc = StandardScaler().fit(xt_tr[:, cols])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(sc.transform(xt_tr[:, cols]), y_tr)
    return sc, clf


def score_with(sc, clf, xt, cols):
    return clf.decision_function(sc.transform(xt[:, cols]))


def direction(sc, clf, cols, V):
    """Back out the activation-space direction the trait weights define."""
    w_traits = np.zeros(V.shape[0])
    w_traits[cols] = clf.coef_.ravel() / sc.scale_
    return unit(w_traits @ V)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified_path",
                    default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    ap.add_argument("--activations_path",
                    default="full_trait_output/harmbench_activations/activations.pt")
    ap.add_argument("--trait_matrix_path", default="full_trait_output/trait_matrix_layer16.npy")
    ap.add_argument("--trait_names_path", default="full_trait_output/trait_names_layer16.json")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--n_cs", type=int, default=60)
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--out", default="results/lasso_sweep.json")
    args = ap.parse_args()

    t0 = time.time()
    V = np.load(args.trait_matrix_path).astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    names = json.loads(Path(args.trait_names_path).read_text())
    n_traits = V.shape[0]

    # Seven activations.pt files dominate the runtime and never change; the trait
    # projections they reduce to are 229-dim, so cache those instead.
    cache = REPO / f"full_trait_output/lasso_cache_layer{args.layer}.npz"
    if cache.exists() and not args.rebuild_cache:
        z = np.load(cache, allow_pickle=True)
        xt, y = z["xt"], z["y"]
        valid = [{"behavior_id": b, "jailbreak_idx": int(t)}
                 for b, t in zip(z["beh"], z["tpl"])]
        tf = {f: (z[f"tf_x_{f}"], z[f"tf_y_{f}"]) for f in z["fams"]}
        print(f"cache {cache.name}: HarmBench n={len(y)} jb={y.mean():.3f}", flush=True)
    else:
        rows = [r for r in load_jsonl(Path(args.classified_path))
                if r.get("attack_type") == "human_jailbreak"]
        acts = torch.load(args.activations_path, map_location="cpu", weights_only=False)
        x, y, valid = build_activation_matrix(rows, acts, args.layer)
        del acts
        xt = x.astype(np.float64) @ V.T
        print(f"HarmBench human_jailbreak {x.shape}  jb={y.mean():.3f}  "
              f"traits={n_traits}", flush=True)
        # transfer families: fixed evaluation sets, never touched by any fit
        tf = {}
        for name, rel in TRANSFER:
            cp = REPO / "full_trait_output" / rel
            ap_ = cp.parent / "activations.pt"
            if not (cp.exists() and ap_.exists()):
                continue
            a = torch.load(ap_, map_location="cpu", weights_only=False)
            xr, yr, _ = build_activation_matrix(load_jsonl(cp), a, args.layer)
            del a
            if len(xr):
                tf[name] = (xr.astype(np.float64) @ V.T, yr)
                print(f"  {name}: n={len(yr)} jb={yr.mean():.3f}", flush=True)
        blob = {"xt": xt, "y": y, "fams": np.array(list(tf)),
                "beh": np.array([r["behavior_id"] for r in valid]),
                "tpl": np.array([r["jailbreak_idx"] for r in valid])}
        for f, (fx, fy) in tf.items():
            blob[f"tf_x_{f}"], blob[f"tf_y_{f}"] = fx, fy
        np.savez_compressed(cache, **blob)
        print(f"  cached -> {cache}", flush=True)
    fams = list(tf)

    cs = np.logspace(-5, 1.5, args.n_cs)
    rec = {"kgrid": KGRID, "families": fams, "names": names, "cs": cs.tolist(), "seeds": []}

    for seed in range(args.n_seeds):
        trb, trt, teb, tet = get_pool_split(valid, args.train_frac, seed)
        tr, te = split_by_pool(valid, trb, trt, teb, tet)
        xtr, ytr, xte, yte = xt[tr], y[tr], xt[te], y[te]
        allc = np.arange(n_traits)

        # dense reference: the published PersonaJB recipe on this split
        sc_d, clf_d = fit_l2(xtr, ytr, allc)
        w_dense = direction(sc_d, clf_d, allc, V)
        dense = {"auc": auc(yte, score_with(sc_d, clf_d, xte, allc)),
                 "transfer": {f: auc(tf[f][1], score_with(sc_d, clf_d, tf[f][0], allc))
                              for f in fams}}

        # L1 path
        sc = StandardScaler().fit(xtr)
        ztr, zte = sc.transform(xtr), sc.transform(xte)
        path = []
        for C in cs:
            clf = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C,
                                     max_iter=5000, random_state=42)
            clf.fit(ztr, ytr)
            coef = clf.coef_.ravel()
            sup = np.flatnonzero(coef)
            e = {"C": float(C), "k": int(sup.size), "support": sup.tolist(),
                 "coef": coef[sup].tolist(),
                 "auc": auc(yte, clf.decision_function(zte)) if sup.size else 0.5,
                 "transfer": {f: (auc(tf[f][1], clf.decision_function(sc.transform(tf[f][0])))
                                  if sup.size else 0.5) for f in fams}}
            if sup.size:
                wt = np.zeros(n_traits)
                wt[sup] = coef[sup] / sc.scale_[sup]
                e["cos_dense"] = float(abs(unit(wt @ V) @ w_dense))
            else:
                e["cos_dense"] = 0.0
            path.append(e)

        # The three selection rules at each budget. All three are scored with the
        # SAME unpenalised L2 estimator on the chosen columns, so the curves differ
        # only in which traits are kept -- the raw L1 path above still carries its
        # own shrunk coefficients, which is a different (and confounded) question.
        uni = np.array([abs(auc(ytr, xtr[:, j]) - 0.5) for j in range(n_traits)])
        order = np.argsort(-uni)
        rng = np.random.default_rng(1000 + seed)
        base = {}
        for k in KGRID:
            out = {}
            hit = [e for e in path if 0 < e["k"] <= k]
            sel = np.array(max(hit, key=lambda e: e["k"])["support"]) if hit else None
            for tag, cols in (("lasso", sel), ("univar", order[:k]),
                              ("random", rng.choice(n_traits, size=k, replace=False))):
                if cols is None:
                    continue
                s_, c_ = fit_l2(xtr, ytr, cols)
                out[tag] = {"auc": auc(yte, score_with(s_, c_, xte, cols)),
                            "transfer": {f: auc(tf[f][1], score_with(s_, c_, tf[f][0], cols))
                                         for f in fams},
                            "cos_dense": float(abs(direction(s_, c_, cols, V) @ w_dense))}
                out[tag]["support"] = [int(i) for i in cols]
            base[str(k)] = out

        rec["seeds"].append({"seed": seed, "n_train": len(tr), "n_test": len(te),
                             "n_test_pos": int(yte.sum()), "dense": dense,
                             "path": path, "budget": base})
        print(f"  seed {seed:2d}  dense {dense['auc']:.3f}  "
              f"k@C=0.01 {path[np.searchsorted(cs, 0.01)]['k']:3d}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rec))
    print(f"wrote {outp}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
