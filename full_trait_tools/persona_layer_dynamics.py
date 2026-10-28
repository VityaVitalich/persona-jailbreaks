#!/usr/bin/env python3
"""
persona_layer_dynamics.py

Per-layer single-trait detection AUC for the top predictive personas (interp L113).
For each persona and each layer L, project the layer-L HarmBench activation onto that
persona's layer-L trait vector and score jailbroken-vs-refused (safe_auc,
orientation-invariant). Uses the all-layer activation dump (layers 0-31) and the
(32, 4096) per-layer trait vectors. Also plots the mean single-trait AUC over all 229
personas per layer, as a baseline for the highlighted ones. Emits results.json and a
line plot (AUC vs layer).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PERSONAS = ["egalitarian", "universalist", "progressive", "principled", "deontological"]
ACTS = "full_trait_output/all_layers_activations/harmbench_activations.pt"
ROWS = "full_trait_output/all_layers_activations/harmbench_rows.jsonl"
VECS = "full_trait_output/paper_data/building_blocks/persona_trait_vectors_llama"


def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    a = float(roc_auc_score(y, s))
    return max(a, 1.0 - a)


def clf_auc(P, y, beh, tmpl, seeds=10, test_frac=0.3):
    """Held-out AUC of a logreg on ALL trait projections at one layer -- the all-traits
    detector we actually use. Matches the paper's protocol: BOTH the behaviors and the
    jailbreak templates of the test set are unseen during training (a random split leaks
    template->success and inflates AUC to ~0.97). Averaged over seeds."""
    if len(np.unique(y)) < 2:
        return float("nan")
    ub, ut = np.unique(beh), np.unique(tmpl)
    aucs = []
    for s in range(seeds):
        rng = np.random.RandomState(s)
        tb = set(rng.choice(ub, int(round(len(ub) * (1 - test_frac))), replace=False))
        tt = set(rng.choice(ut, max(1, int(round(len(ut) * (1 - test_frac)))), replace=False))
        trm = np.array([(b in tb) and (t in tt) for b, t in zip(beh, tmpl)])
        tem = np.array([(b not in tb) and (t not in tt) for b, t in zip(beh, tmpl)])
        if trm.sum() < 50 or tem.sum() < 20 or len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            continue
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000))]).fit(P[trm], y[trm])
        aucs.append(roc_auc_score(y[tem], pipe.decision_function(P[tem])))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=PERSONAS)
    ap.add_argument("--output_dir", default="full_trait_output/persona_layer_dynamics")
    ap.add_argument("--attack_type", default="human_jailbreak",
                    help="keep only rows with this attack_type (matches the paper's HarmBench set)")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    label, beh_of, tmpl_of = {}, {}, {}
    for r in rows:
        if args.attack_type and r.get("attack_type") != args.attack_type:
            continue
        pid = r.get("pair_id")
        jb = r.get("jailbroken", r.get("binary_jailbroken"))
        if pid is not None and jb is not None:
            label[str(pid)] = int(bool(jb))
            beh_of[str(pid)] = r.get("behavior_id", str(pid))
            tmpl_of[str(pid)] = r.get("jailbreak_idx", -1)

    acts = torch.load(ACTS, map_location="cpu", weights_only=False)
    keys = [k for k in acts if str(k) in label]
    layers = sorted((int(x) for x in acts[keys[0]].keys()))
    y = np.array([label[str(k)] for k in keys])
    groups = np.array([beh_of[str(k)] for k in keys])
    tmpl = np.array([tmpl_of[str(k)] for k in keys])
    print(f"n_pairs={len(keys)}, jb_rate={y.mean():.3f}, layers={layers[0]}..{layers[-1]}", flush=True)

    # Load all 229 trait vectors: names + [n, 32, 4096]
    names, vecs = [], []
    for f in sorted(glob.glob(f"{VECS}/*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        names.append(d.get("trait", os.path.basename(f)[:-3]))
        vecs.append(d["vector"].float().numpy())
    V = np.stack(vecs)  # [n, 32, 4096]
    n_traits = V.shape[0]
    print(f"loaded {n_traits} trait vectors", flush=True)

    # Per-layer trait projections + single-trait AUCs
    auc_by_layer = np.full((len(layers), n_traits), np.nan)  # [L, n_traits]
    P_layers = []
    for li, L in enumerate(layers):
        XL = np.stack([acts[k][str(L)].float().numpy() for k in keys])  # [n_pairs, 4096]
        P = (XL @ V[:, L, :].T).astype(np.float32)  # [n_pairs, n_traits]
        P_layers.append(P)
        for t in range(n_traits):
            auc_by_layer[li, t] = safe_auc(y, P[:, t])
        print(f"  projected L{L}", flush=True)

    # All-traits classifier (CV logreg) AUC per layer, parallel over layers
    from joblib import Parallel, delayed
    all_traits = np.array(Parallel(n_jobs=16)(delayed(clf_auc)(P_layers[li], y, groups, tmpl)
                                              for li in range(len(layers))))
    for li, L in enumerate(layers):
        print(f"  L{L:2d}: all-traits clf AUC={all_traits[li]:.3f}", flush=True)

    mean_all = np.nanmean(auc_by_layer, axis=1)  # [L]
    name_idx = {nm: names.index(nm) for nm in args.personas}

    results = {nm: [round(float(auc_by_layer[li, name_idx[nm]]), 4) for li in range(len(layers))]
               for nm in args.personas}
    results["mean_all_229"] = [round(float(v), 4) for v in mean_all]
    results["all_traits_clf"] = [round(float(v), 4) for v in all_traits]

    for nm in args.personas:
        a = results[nm]
        peak = int(np.nanargmax(a))
        print(f"  {nm:14s} peak {max(a):.3f} @ L{layers[peak]} | L16={a[layers.index(16)]:.3f}", flush=True)
    print(f"  mean(all 229)  peak {np.nanmax(mean_all):.3f} @ L{layers[int(np.nanargmax(mean_all))]} "
          f"| L16={mean_all[layers.index(16)]:.3f}", flush=True)
    print(f"  all-traits clf peak {np.nanmax(all_traits):.3f} @ L{layers[int(np.nanargmax(all_traits))]} "
          f"| L16={all_traits[layers.index(16)]:.3f}", flush=True)

    payload = {"metric": "per_layer_single_trait_auc", "layers": layers,
               "n_pairs": len(keys), "jb_rate": float(y.mean()), "n_traits": n_traits,
               "results": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for nm in args.personas:
            ax.plot(layers, results[nm], marker="o", ms=3, label=nm)
        ax.plot(layers, results["all_traits_clf"], color="crimson", lw=3.0,
                label="all-229 trait classifier (logreg)")
        ax.plot(layers, results["mean_all_229"], color="black", lw=2.4, ls="--",
                label="mean single trait (all 229)")
        ax.axhline(0.5, ls="--", c="gray", lw=0.8, label="chance")
        ax.axvline(16, ls=":", c="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("layer"); ax.set_ylabel("detection AUC (HarmBench)")
        ax.set_title("Per-layer persona predictiveness (HarmBench)")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "persona_layer_dynamics.png", dpi=200)
        print(f"Saved plot to {out / 'persona_layer_dynamics.png'}", flush=True)
    except Exception as e:
        print(f"[plot skipped] {e}", flush=True)


if __name__ == "__main__":
    main()
