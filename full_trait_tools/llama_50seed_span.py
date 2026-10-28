#!/usr/bin/env python3
"""50-seed Llama trait-span decomposition: the numbers behind
fig_direction_story and appendix Tables tab:subspace-projection /
tab:refusal-direction.

50-seed extension of trait_span_and_weights.py. Same protocol as the published
single-split tables -- human_jailbreak rows only, behaviour- and
template-disjoint pool splits (get_pool_split, train_frac 0.7), HarmBench only,
layer 16 -- but averaged over 50 splits instead of reported at seed 0, because
a single split of this pool swings by ~0.07 AUC (see the per-seed std) and the
seed-0 values are not reproducible from any recorded run.

For each direction: share of squared norm inside the 229-trait span, and
held-out jailbreak-detection AUC of the direction, of its in-span component
P d, and of its residual (I-P) d.

Directions:
  refusal_pos1  Arditi harmful-vs-harmless diff-of-means, layer 16, last token
  refusal_pos5  the same at position -5, which Arditi's causal selection
                prefers at this layer
  dmu           jailbroken-minus-refused class means, refit on each train split
  w_raw         the raw-activation logistic-regression normal (a fixed saved
                vector, not refit)
  random        20 random unit directions, for the AUC floor

Seed 0 does NOT reproduce the values the tables used to print, and no other
seed does either. It gives refusal_pos1 0.812/0.824 (published 0.818/0.842),
dmu 0.808/0.787 share .588 (published 0.760/0.721/.446) and w_raw 0.919/0.869
(published 0.839/0.841), on n_test 288 rather than the 252 the old caption
claimed. The split-free shares do match (.407, .087), so the direction and the
trait span are right and it is the split that differs; the producing run is
lost. That is the reason for averaging rather than picking a seed, and the
reason w_raw is no longer reported in the paper: its 0.919 here is far above
the published 0.839 and the saved vector's training set is not recorded, so it
cannot be shown to be a held-out number.

Usage: python3.13 full_trait_tools/llama_50seed_span.py [--out PATH]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "full_trait_tools"))
from run_all_traits_sweep_v2 import (            # noqa: E402
    build_activation_matrix, get_pool_split, load_jsonl, safe_auc,
    split_by_pool,
)

N_SEEDS, TRAIN_FRAC, LAYER, N_RANDOM = 50, 0.7, 16, 20
REF_DIR = Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct"
# what the tables used to print at "seed 0"; none of it reproduces (see module
# docstring) -- kept so the run prints the discrepancy rather than hiding it
PUBLISHED = {"refusal_pos1": (0.818, 0.842, 0.407),
             "dmu": (0.760, 0.721, 0.446),
             "w_raw": (0.839, 0.841, 0.087)}


def unit(v):
    return v / np.linalg.norm(v)


def load_direction(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(o, dict):
        o = o.get("direction", o.get("vector", o.get("w", o.get("normal"))))
    return unit(np.asarray(o, dtype=np.float64).ravel())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO / "results/llama_50seed_span.json"))
    a = p.parse_args()

    rows = [r for r in load_jsonl(
        REPO / "full_trait_output/harmbench_activations/classified_responses.jsonl")
        if r.get("attack_type") == "human_jailbreak"]
    acts = torch.load(REPO / "full_trait_output/harmbench_activations/activations.pt",
                      map_location="cpu", weights_only=False)
    x_raw, y, valid = build_activation_matrix(rows, acts, LAYER)
    V = pickle.load(open(REPO / "full_trait_output/trait_detector.pkl", "rb"))["trait_matrix"]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    Q, _ = np.linalg.qr(V.T.astype(np.float64))           # [hidden, rank]
    print(f"acts {x_raw.shape}, traits {V.shape}, jb rate {y.mean():.3f}", flush=True)

    fixed = {
        "refusal_pos1": load_direction(REF_DIR / "direction_layer16_lasttoken.pt"),
        "refusal_pos5": load_direction(REF_DIR / "direction_layer16.pt"),
        "w_raw": load_direction(
            REPO / "full_trait_output/harmbench_logreg/hyperplane_normal_layer16.pt"),
    }
    # the refusal direction points toward refusal, so it is scored negated to
    # give a jailbreak-positive AUC; w_raw and dmu already point at jailbreak
    sign = {"refusal_pos1": -1.0, "refusal_pos5": -1.0, "w_raw": 1.0, "dmu": 1.0}

    def decompose(d):
        pd_ = Q @ (Q.T @ d)
        rd = d - pd_
        return pd_, rd, float(pd_ @ pd_ / (d @ d))

    names = list(fixed) + ["dmu"]
    res = {n: {"full": [], "proj": [], "resid": []} for n in names}
    res["dmu"]["share"] = []
    shares = {n: decompose(v)[2] for n, v in fixed.items()}
    # the figure quotes dmu's share on all data, not the per-split mean
    dmu_all = unit(x_raw[y == 1].mean(0) - x_raw[y == 0].mean(0)).astype(np.float64)
    shares["dmu_alldata"] = decompose(dmu_all)[2]
    res["random"] = {"full": [], "share": []}
    res["n_test"], res["n_test_jb"] = [], []

    rng = np.random.default_rng(0)
    for seed in range(N_SEEDS):
        tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(valid, TRAIN_FRAC, seed)
        tr, te = split_by_pool(valid, tr_beh, tr_tpl, te_beh, te_tpl)
        xt, yt = x_raw[te].astype(np.float64), y[te]

        def auc(d, n):
            return safe_auc(yt, sign[n] * (xt @ unit(d)))

        for n, d in fixed.items():
            pd_, rd, _ = decompose(d)
            res[n]["full"].append(auc(d, n))
            res[n]["proj"].append(auc(pd_, n))
            res[n]["resid"].append(auc(rd, n))
        dmu = unit((x_raw[tr][y[tr] == 1].mean(0)
                    - x_raw[tr][y[tr] == 0].mean(0)).astype(np.float64))
        pd_, rd, sh = decompose(dmu)
        res["dmu"]["full"].append(auc(dmu, "dmu"))
        res["dmu"]["proj"].append(auc(pd_, "dmu"))
        res["dmu"]["resid"].append(auc(rd, "dmu"))
        res["dmu"]["share"].append(sh)
        res["n_test"].append(len(te))
        res["n_test_jb"].append(int(yt.sum()))
        if seed == 0:
            for n, (f0, p0, s0) in PUBLISHED.items():
                sh0 = res["dmu"]["share"][0] if n == "dmu" else shares[n]
                print(f"  seed0 {n:12s} {res[n]['full'][0]:.3f}/{res[n]['proj'][0]:.3f} "
                      f"share {sh0:.3f}   published {f0}/{p0}/{s0}", flush=True)
        for _ in range(N_RANDOM // 10):     # 2 per seed -> 100 draws overall
            d = unit(rng.standard_normal(x_raw.shape[1]))
            # plain AUC here, not safe_auc: the sign-agnostic max(auc, 1-auc)
            # folds a random direction's two tails onto one side and reports
            # the floor as 0.64 instead of 0.5
            res["random"]["full"].append(float(roc_auc_score(yt, xt @ d)))
            res["random"]["share"].append(decompose(d)[2])

    summ = {"shares": shares, "n_test_median": float(np.median(res["n_test"])),
            "n_test_jb_median": float(np.median(res["n_test_jb"])),
            "n_seeds": N_SEEDS}
    for n in names + ["random"]:
        for k, v in res[n].items():
            arr = np.array(v, dtype=float)
            summ[f"{n}_{k}"] = {"mean": float(arr.mean()), "std": float(arr.std()),
                                "ci95_mean": float(1.96 * arr.std() / np.sqrt(len(arr)))}
        if n != "random":
            diff = np.array(res[n]["proj"]) - np.array(res[n]["full"])
            summ[f"{n}_paired_diff"] = {
                "mean": float(diff.mean()), "std": float(diff.std()),
                "ci95_mean": float(1.96 * diff.std() / np.sqrt(N_SEEDS))}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summ, "per_seed": res}, indent=1))

    print(f"\n{'direction':14} {'share':>7} {'AUC full':>18} {'AUC proj':>18} "
          f"{'AUC resid':>18}")
    for n in names:
        sh = shares.get(n, summ.get("dmu_share", {}).get("mean"))
        cells = []
        for k in ("full", "proj", "resid"):
            s = summ[f"{n}_{k}"]
            cells.append(f"{s['mean']:.3f} +/- {s['ci95_mean']:.3f}")
        print(f"{n:14} {sh:7.3f} " + " ".join(f"{c:>18}" for c in cells))
    r = summ["random_full"]
    print(f"{'random':14} {summ['random_share']['mean']:7.3f} "
          f"{r['mean']:.3f} +/- {r['std']:.3f} (std over {len(res['random']['full'])} draws)")
    print(f"\ndmu share: per-split {summ['dmu_share']['mean']:.3f}, "
          f"all data {shares['dmu_alldata']:.3f}")
    print(f"n_test median {summ['n_test_median']:.0f} "
          f"({summ['n_test_jb_median']:.0f} jailbroken)")
    print("saved", out)


if __name__ == "__main__":
    main()
