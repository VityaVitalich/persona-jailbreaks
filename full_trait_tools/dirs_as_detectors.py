"""The refusal direction and the jailbreak direction, as jailbreak detectors.

Both exist for both models; what has never been measured is their transfer to
the five unseen attack families. The direction-story figure reports only
in-distribution HarmBench AUC (Refusal .844/.799, Jailbreak .856/.861).

Vectors
  Refusal, Llama  ~/refusal_direction/runs/llama-3.1-8b-instruct/
                  direction_layer16{,_lasttoken}.pt (Arditi recreation)
  Refusal, Gemma  Arditi diff-of-means rebuilt from the class means in
                  moskvore/gemma_refusal_dir: means_harmful - means_harmless,
                  [61 layers, 5 positions, 5376]; meta.json fixes the
                  convention as hidden_states[k], trait-space layer 30 ==
                  hidden_states[31], so pos -1 is index [31, 4]. Both the
                  unfiltered and the filtered extraction are checked against
                  the figure's published in-span share to identify which one
                  the paper used.
  Jailbreak       mean(jailbroken) - mean(refused), refit per seed.

Protocol. The refusal direction is estimated on Arditi's own harmful/harmless
instructions, so nothing of ours is in it: HarmBench and all five families are
held out and there is no training pool to match. The jailbreak direction is
fitted on our labels, so it gets what every trained row gets -- re-estimated on
the strict 70/30 behaviour x template train pool, 50 seeds, evaluated on the
disjoint HarmBench test cell and the families. Gemma folds in WildJailbreak,
as its other trained rows do.

Identity check: each direction's share of squared norm inside the trait span
must reproduce the published figure (Llama refusal .407, jailbreak .482;
Gemma refusal .265, jailbreak .330).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_traits_sweep_v2 import (  # noqa: E402
    build_activation_matrix, get_pool_split, load_jsonl, split_by_pool)

FT = REPO / "full_trait_output"
FAMS = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
CFG = {
    "Llama-3.1-8B": dict(
        layer=16, hb="harmbench_activations", fam_file="responses.jsonl",
        fam={f: f"{f.lower()}_activations" for f in FAMS}, augment=None,
        traits="trait_matrix_layer16.npy",
        published_share={"refusal": 0.407, "jailbreak": 0.482},
        ref=("PersonaJB Direction", 0.854,
             [0.8166, 0.7900, 0.6413, 0.7083, 0.8416])),
    "Gemma-4-31B": dict(
        layer=30, hb="harmbench_activations_gemma",
        fam_file="classified_responses.jsonl",
        fam={f: f"{f.lower()}_activations_gemma_hb" for f in FAMS},
        augment="wildjailbreak_activations_gemma",
        traits="gemma_sweep_trait/trait_matrix_layer30.npy",
        published_share={"refusal": 0.265, "jailbreak": 0.330},
        ref=("PersonaJB Direction", 0.8773,
             [0.7383, 0.6782, 0.6937, 0.7590, 0.8402])),
}


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / np.linalg.norm(v)


def flip(a):
    return max(a, 1.0 - a)


def auc(y, s):
    return float(roc_auc_score(y, s))


def load(d, f, layer, filt=False):
    rows = load_jsonl(FT / d / f)
    if filt:
        rows = [r for r in rows if r.get("attack_type") == "human_jailbreak"]
    a = torch.load(FT / d / "activations.pt", map_location="cpu", weights_only=False)
    x, y, valid = build_activation_matrix(rows, a, layer)
    return x.astype(np.float64), y, valid


def gemma_refusal(variant, pos):
    """Arditi diff-of-means at trait-space layer 30 (= hidden_states[31])."""
    B = FT / "gemma_refusal_dir"
    if variant == "unfiltered":
        hf, hl = "out/means_harmful.pt", "out/means_harmless.pt"
    else:
        hf, hl = ("out_filtered/means_harmful_filt.pt",
                  "out_filtered/means_harmless_filt.pt")
    a = torch.load(B / hf, map_location="cpu", weights_only=False)[31, pos]
    b = torch.load(B / hl, map_location="cpu", weights_only=False)[31, pos]
    return unit((a - b).float().numpy())


def llama_refusal(pos1):
    p = Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct"
    f = "direction_layer16_lasttoken.pt" if pos1 else "direction_layer16.pt"
    d = torch.load(p / f, map_location="cpu", weights_only=False)
    return unit(d["direction"].numpy())


for model, cfg in CFG.items():
    L = cfg["layer"]
    x_hb, y_hb, valid = load(cfg["hb"], "classified_responses.jsonl", L, filt=True)
    fam = {f: load(cfg["fam"][f], cfg["fam_file"], L)[:2] for f in FAMS}
    aug = load(cfg["augment"], "classified_responses.jsonl", L)[:2] if cfg["augment"] else None

    V = np.load(FT / cfg["traits"]).astype(np.float64)
    Q, _ = np.linalg.qr(V.T)
    span = lambda d: float((Q @ (Q.T @ d)) @ (Q @ (Q.T @ d)) / (d @ d))  # noqa: E731

    print(f"\n{'='*100}\n{model}  ({len(y_hb)} HarmBench pairs, "
          f"{V.shape[0]}-trait span)\n{'='*100}")

    # --- identify the Gemma refusal extraction by its published span share ---
    if model.startswith("Gemma"):
        for variant in ("unfiltered", "filtered"):
            d = gemma_refusal(variant, 4)
            print(f"  refusal[{variant:10s}] pos -1: in-span {span(d):.3f} "
                  f"(published {cfg['published_share']['refusal']:.3f})")
        refusals = [("Refusal dir (pos -1)", gemma_refusal("filtered", 4)),
                    ("Refusal dir (pos -5)", gemma_refusal("filtered", 0))]
    else:
        refusals = [("Refusal dir (pos -1)", llama_refusal(True)),
                    ("Refusal dir (pos -5)", llama_refusal(False))]

    print(f"\n  {'direction':24s} {'span':>5s} {'HarmB':>6s} "
          + "".join(f"{f:>8s}" for f in FAMS) + f" {'AvgXfer':>8s}  note")

    name, ref_hb, ref_per = cfg["ref"]
    print(f"  {name+' (ref)':24s} {1.000:5.3f} {ref_hb:6.3f} "
          + "".join(f"{v:8.3f}" for v in ref_per)
          + f" {statistics.fmean(ref_per):8.3f}  70/30 + C=10, 50 seeds")

    # --- refusal: fixed, zero-shot on everything of ours; scored negated ---
    for lbl, r in refusals:
        d = -r                      # r points toward refusal
        per = [auc(fam[f][1], fam[f][0] @ d) for f in FAMS]
        hb = auc(y_hb, x_hb @ d)
        print(f"  {lbl:24s} {span(d):5.3f} {flip(hb):6.3f} "
              + "".join(f"{flip(v):8.3f}" for v in per)
              + f" {statistics.fmean(flip(v) for v in per):8.3f}  zero-shot (Arditi data)")
        if any(v < 0.5 for v in per + [hb]):
            print(f"  {'':24s} {'':5s} signed: {hb:.3f} "
                  + "".join(f"{v:8.3f}" for v in per))

    # --- jailbreak direction: refit per seed on the matched train pool ---
    per_seed = {k: [] for k in ["HarmBench"] + FAMS}
    shares = []
    for seed in range(50):
        tb, tt, eb, et = get_pool_split(valid, 0.7, seed)
        tr, te = split_by_pool(valid, tb, tt, eb, et)
        if len(np.unique(y_hb[tr])) < 2 or len(np.unique(y_hb[te])) < 2:
            continue
        xt, yt = x_hb[tr], y_hb[tr]
        if aug is not None:
            xt, yt = np.vstack([xt, aug[0]]), np.concatenate([yt, aug[1]])
        d = unit(xt[yt == 1].mean(0) - xt[yt == 0].mean(0))
        shares.append(span(d))
        per_seed["HarmBench"].append(auc(y_hb[te], x_hb[te] @ d))
        for f in FAMS:
            per_seed[f].append(auc(fam[f][1], fam[f][0] @ d))
    n = len(per_seed["HarmBench"])
    xfer = [statistics.fmean(flip(per_seed[f][i]) for f in FAMS) for i in range(n)]
    ci = 1.96 * statistics.stdev(xfer) / n ** 0.5
    tag = "train pool + WJB" if aug is not None else "train pool"
    print(f"  {'Jailbreak dir (dmu)':24s} {statistics.fmean(shares):5.3f} "
          f"{flip(statistics.fmean(per_seed['HarmBench'])):6.3f} "
          + "".join(f"{flip(statistics.fmean(per_seed[f])):8.3f}" for f in FAMS)
          + f" {statistics.fmean(xfer):8.3f}  {tag}, {n} seeds, +-{ci:.3f}")
    print(f"  {'':24s} {'':5s} published in-span share "
          f"{cfg['published_share']['jailbreak']:.3f}, "
          f"signed per family: "
          + " ".join(f"{statistics.fmean(per_seed[f]):.3f}" for f in FAMS))
