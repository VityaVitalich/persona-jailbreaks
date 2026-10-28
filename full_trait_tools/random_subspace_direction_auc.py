#!/usr/bin/env python3
"""Matched random-subspace controls for the direction-story figure.

For Llama-3.1-8B and Gemma-4-31B, evaluate the refusal direction and the
jailbreak class-mean direction on up to 50 behaviour- and template-disjoint
HarmBench splits used by the paper figure.  Compare each full direction and its
persona-span projection with its projection onto independently sampled random
subspaces of the same dimensionality as that model's persona span.

The random-subspace unit is the replicate: for each sampled subspace, AUC is
averaged over the valid held-out splits.  Report mean +/- standard deviation over
random subspaces.  Gemma's jailbreak direction uses the paper protocol in which
WildJailbreak rows augment the training pool only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "full_trait_tools"))
from run_all_traits_sweep_v2 import (  # noqa: E402
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    split_by_pool,
)

FT = REPO / "full_trait_output"
N_SEEDS = 50
TRAIN_FRAC = 0.7


def unit_columns(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=0, keepdims=True)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    # Direction signs are fixed before evaluation. Do not flip each test split:
    # doing so would bias weak random-subspace projections above chance.
    return float(roc_auc_score(y, score))


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # NumPy 2.4 on macOS can leave floating-point status flags set after an
    # otherwise finite Accelerate matmul. Check the result directly.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = a @ b
    if not np.isfinite(value).all():
        raise FloatingPointError("non-finite matrix product")
    return value


def load_activation_pool(directory: Path, layer: int, human_only: bool = False):
    rows = load_jsonl(directory / "classified_responses.jsonl")
    if human_only:
        rows = [r for r in rows if r.get("attack_type") == "human_jailbreak"]
    acts = torch.load(directory / "activations.pt", map_location="cpu", weights_only=False)
    x, y, valid = build_activation_matrix(rows, acts, layer)
    return x.astype(np.float64), y, valid


def load_refusal_direction(model: str) -> np.ndarray:
    if model == "llama":
        obj = torch.load(
            Path.home()
            / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt",
            map_location="cpu",
            weights_only=False,
        )
        refusal = obj["direction"].numpy()
    elif model == "gemma":
        base = FT / "gemma_refusal_dir/out"
        harmful = torch.load(base / "means_harmful.pt", map_location="cpu", weights_only=False)
        harmless = torch.load(base / "means_harmless.pt", map_location="cpu", weights_only=False)
        refusal = (harmful[31, 4] - harmless[31, 4]).float().numpy()
    else:
        raise ValueError(model)
    # Score toward jailbreak, matching the direction-story figure.
    return unit_columns((-np.asarray(refusal, dtype=np.float64)).reshape(-1, 1))[:, 0]


def load_persona_matrix(model: str) -> np.ndarray:
    if model == "llama":
        with open(FT / "trait_detector.pkl", "rb") as f:
            matrix = pickle.load(f)["trait_matrix"]
    elif model == "gemma":
        matrix = np.load(FT / "gemma_sweep_trait/trait_matrix_layer30.npy")
    else:
        raise ValueError(model)
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(len(values)),
    }


def evaluate_model(model: str, n_random: int, random_seed: int) -> dict:
    if model == "llama":
        layer = 16
        hb_dir = FT / "harmbench_activations"
        aug = None
    elif model == "gemma":
        layer = 30
        hb_dir = FT / "harmbench_activations_gemma"
        aug = load_activation_pool(FT / "wildjailbreak_activations_gemma", layer)[:2]
    else:
        raise ValueError(model)

    x, y, valid = load_activation_pool(hb_dir, layer, human_only=True)
    traits = load_persona_matrix(model)
    q_persona, _ = np.linalg.qr(traits.T, mode="reduced")
    hidden, span_dim = q_persona.shape
    refusal = load_refusal_direction(model)
    if refusal.shape != (hidden,):
        raise ValueError(f"{model}: refusal {refusal.shape}, hidden size {hidden}")

    test_indices: list[np.ndarray] = []
    jailbreak_directions: list[np.ndarray] = []
    seeds_used: list[int] = []
    for seed in range(N_SEEDS):
        tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(valid, TRAIN_FRAC, seed)
        tr, te = split_by_pool(valid, tr_beh, tr_tpl, te_beh, te_tpl)
        tr = np.asarray(tr, dtype=int)
        te = np.asarray(te, dtype=int)
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2:
            continue
        if aug is None:
            x_fit, y_fit = x[tr], y[tr]
        else:
            x_fit = np.vstack([x[tr], aug[0]])
            y_fit = np.concatenate([y[tr], aug[1]])
        dmu = x_fit[y_fit == 1].mean(0) - x_fit[y_fit == 0].mean(0)
        jailbreak_directions.append(dmu / np.linalg.norm(dmu))
        test_indices.append(te)
        seeds_used.append(seed)

    dmu_matrix = np.column_stack(jailbreak_directions)
    n_splits = len(seeds_used)
    directions = {
        "refusal": np.repeat(refusal[:, None], n_splits, axis=1),
        "jailbreak": dmu_matrix,
    }

    result = {
        "model": model,
        "layer": layer,
        "hidden_dim": hidden,
        "persona_span_dim": span_dim,
        "n_splits": n_splits,
        "seeds_used": seeds_used,
        "n_random_subspaces": n_random,
        "random_seed": random_seed,
        "directions": {},
    }

    # Full and persona-projected scores, one direction per held-out split.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        persona_projected = matmul(q_persona, matmul(q_persona.T, np.column_stack(
            [directions["refusal"][:, 0], *jailbreak_directions]
        )))
    if not np.isfinite(persona_projected).all():
        raise FloatingPointError(f"{model}: non-finite persona projection")
    persona_refusal = persona_projected[:, 0]
    persona_dmu = persona_projected[:, 1:]

    for name, dmat in directions.items():
        full_aucs = []
        persona_aucs = []
        persona_shares = []
        pmat = persona_dmu if name == "jailbreak" else np.repeat(
            persona_refusal[:, None], n_splits, axis=1
        )
        for seed, te in enumerate(test_indices):
            full_aucs.append(auc(y[te], matmul(x[te], dmat[:, seed])))
            persona_aucs.append(auc(y[te], matmul(x[te], pmat[:, seed])))
            persona_shares.append(float(np.dot(pmat[:, seed], pmat[:, seed])))
        result["directions"][name] = {
            "full_auc_over_splits": summarize(np.array(full_aucs)),
            "persona_auc_over_splits": summarize(np.array(persona_aucs)),
            "persona_share_over_splits": summarize(np.array(persona_shares)),
            "random_subspace_mean_auc": [],
            "random_subspace_mean_share": [],
        }

    rng = np.random.default_rng(random_seed)
    for replicate in range(n_random):
        gaussian = rng.standard_normal((hidden, span_dim))
        q_random, _ = np.linalg.qr(gaussian, mode="reduced")
        del gaussian
        print(f"  {model}: random subspace {replicate + 1}/{n_random}", flush=True)

        for name, dmat in directions.items():
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                projected = matmul(q_random, matmul(q_random.T, dmat))
            if not np.isfinite(projected).all():
                raise FloatingPointError(
                    f"{model}: non-finite random projection at replicate {replicate}"
                )
            aucs = []
            shares = []
            for seed, te in enumerate(test_indices):
                aucs.append(auc(y[te], matmul(x[te], projected[:, seed])))
                shares.append(float(np.dot(projected[:, seed], projected[:, seed])))
            result["directions"][name]["random_subspace_mean_auc"].append(float(np.mean(aucs)))
            result["directions"][name]["random_subspace_mean_share"].append(float(np.mean(shares)))

    for direction_result in result["directions"].values():
        random_aucs = np.array(direction_result.pop("random_subspace_mean_auc"))
        random_shares = np.array(direction_result.pop("random_subspace_mean_share"))
        direction_result["random_auc_per_subspace"] = random_aucs.tolist()
        direction_result["random_share_per_subspace"] = random_shares.tolist()
        direction_result["random_auc_across_subspaces"] = summarize(
            random_aucs
        )
        direction_result["random_share_across_subspaces"] = summarize(
            random_shares
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_random", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=20260914)
    parser.add_argument(
        "--out", default=str(REPO / "results/random_subspace_direction_auc.json")
    )
    args = parser.parse_args()

    results = {
        "protocol": (
            "AUC of refusal and split-fitted jailbreak directions after projection "
            "onto matched-dimensional persona or random subspaces; random replicate "
            "AUCs are means over the same 50 held-out HarmBench splits"
        ),
        "models": {},
    }
    for model in ("llama", "gemma"):
        print(f"\n=== {model} ===", flush=True)
        results["models"][model] = evaluate_model(model, args.n_random, args.random_seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
    for model, model_result in results["models"].items():
        for name, values in model_result["directions"].items():
            full = values["full_auc_over_splits"]["mean"]
            persona = values["persona_auc_over_splits"]["mean"]
            random = values["random_auc_across_subspaces"]
            print(
                f"{model:5s} {name:9s}: full={full:.3f} persona={persona:.3f} "
                f"random={random['mean']:.3f} +/- {random['std']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
