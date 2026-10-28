"""PersonaJB vs Gaussian control vs raw-activation probe, per training set, Llama-3.1-8B / Gemma-4-31B / Qwen3-32B / OLMo-3-7B.
PersonaJB = StandardScaler + LR on projections onto the model's unit trait vectors (pre-generation last token):
  Llama 229 (paper filter; saved fits), Qwen 240 (Assistant Axis release; saved fits), Gemma 240 (paper vectors, layer 30,
  gemma_data/trait_vectors/*.pt row 0), OLMo 240 unfiltered (Dominic's traits40_vectors_olmo3_7b, row 16 = layers[16]
  output = our readout) and, as a variant, the 228 OLMo traits passing the Llama paper filter
  (filter_matched_pairs_ge_50_count_ge_10_total).
Same protocol as four_models.py (C tuned on a behaviour-grouped validation split of the training set; transfer = mean folded
AUC over the other attack families). Gemma / OLMo PersonaJB are fitted here; raw and Gaussian scores come from
scores_four_models.npz; Llama / Qwen everything from scores_cross_family_tuned.npz.
Cluster bootstrap over HarmBench behaviours (B=1000, same draws as four_models.py); paired deltas PersonaJB - Gaussian and
PersonaJB - raw, Bonferroni over (7 training sets + avg) x 4 models x 2 contrasts.
Saves fitted val/test scores and every bootstrap draw to scores_persona_four.npz; table persona_four.json."""
import json, glob, os, numpy as np, torch
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import olmo_probe_lib as L
from olmo_probe_lib import gemma, olmo
SETS, FAM, G = L.SETS, L.FAM, 3
O = "/Users/viktor/persona-vector-jbing/full_trait_output/olmo3_7b"
def tv(paths, row):
    names = sorted(os.path.basename(p)[:-3] for p in paths); d = os.path.dirname(paths[0])
    T = np.stack([torch.load(f"{d}/{n}.pt", map_location="cpu", weights_only=False)["vector"][row].float().numpy() for n in names]).astype(np.float64)
    return T / np.linalg.norm(T, axis=1, keepdims=True), names
To, on = tv(glob.glob(f"{O}/trait_vectors/all_traits_no_filter/*.pt"), 16)
keep = set(open(f"{O}/trait_vectors/filter_matched_pairs_ge_50_count_ge_10_total.txt").read().split())
Tf = To[[i for i, n in enumerate(on) if n in keep]]
Tg, gn = tv(glob.glob(f"{O}/gemma_data/trait_vectors/*.pt"), 0)
print("trait matrices: olmo", To.shape, "olmo filtered", Tf.shape, "gemma", Tg.shape, flush=True)
FZ = np.load("scores_four_models.npz"); CZ = np.load("/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b/transfer/diag/scores_cross_family_tuned.npz")
def get(Z, m, det, a, b): return Z[[k for k in Z.files if k.startswith(f"{m}__{det}__train_{a}__test_{b}__C")][0]]
M, SC = {}, {}
for m, loader, mats in (("gemma", gemma, {"PersonaJB": Tg}), ("olmo", olmo, {"PersonaJB": To, "PersonaJB_filtered": Tf})):
    D = {s: loader(s) for s in SETS}
    jobs = [delayed(L.cell)((det, a), a, D[a][0] @ T.T, D[a][1], D[a][2], {b: D[b][0] @ T.T for b in SETS if b != a}) for det, T in mats.items() for a in SETS]
    OUT = dict(Parallel(n_jobs=8, backend="loky")(jobs)); print(m, "PersonaJB fits done", flush=True)
    assert all((FZ[f"{m}__labels_{s}"] == D[s][1]).all() for s in SETS), "row order differs from four_models"
    M[m] = {"y": {s: D[s][1] for s in SETS}, "beh": {s: FZ[f"{m}__behavior_ids_{s}"] for s in SETS},
            **{det: {(a, b): OUT[((det, a), a)]["test"][b] for a in SETS for b in SETS if a != b} for det in mats},
            **{det: {(a, b): get(FZ, m, det, a, b) for a in SETS for b in SETS if a != b} for det in ["raw"] + [f"Gaussian{g}" for g in range(G)]},
            "C": {f"{det}__{a}": OUT[((det, a), a)]["C"] for det in mats for a in SETS}}
    for ((det, a), _), v in OUT.items():
        p = f"{m}__{det}__train_{a}"; SC[f"{p}__val_idx"] = v["val_idx"]
        for C, s in v["val_scores"].items(): SC[f"{p}__val_C{C:g}"] = s
        for b, s in v["test"].items(): SC[f"{p}__test_{b}__C{v['C']:g}"] = s
for m in ("llama", "qwen"):
    M[m] = {"y": {s: CZ[f"{m}__labels_{s}"] for s in SETS}, "beh": {s: CZ[f"{m}__behavior_ids_{s}"] for s in SETS},
            **{det: {(a, b): get(CZ, m, cd, a, b) for a in SETS for b in SETS if a != b}
               for det, cd in [("PersonaJB", "PersonaJB"), ("raw", "fullprobe")] + [(f"Gaussian{g}", f"Gaussian{g}") for g in range(G)]}}
B = 1000; NB = (len(SETS) + 1) * len(M) * 2; zq = 1 - 0.05 / NB / 2
res = {"sets": SETS, "B": B, "bonferroni_m": NB, "trait_counts": {"llama": 229, "qwen": 240, "gemma": len(gn), "olmo": len(on), "olmo_filtered": len(Tf)},
       "cells": {}, "per_family": {}, "C": {m: M[m].get("C") for m in M}}
ci = lambda b: [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
dci = lambda b: [float(np.nanpercentile(b, 100 * (1 - zq))), float(np.nanpercentile(b, 100 * zq))]
for m, V in M.items():
    behs = sorted({b for s in SETS for b in V["beh"][s]}); bi = {b: i for i, b in enumerate(behs)}
    W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B); bidx = {s: np.array([bi[b] for b in V["beh"][s]]) for s in SETS}
    def boot(y, s, bo):
        sign = 1 if roc_auc_score(y, s) >= 0.5 else -1
        return roc_auc_score(y, sign * s), np.array([roc_auc_score(y, sign * s, sample_weight=W[k][bo]) if (W[k][bo] * y).sum() > 0 and (W[k][bo] * (1 - y)).sum() > 0 else np.nan for k in range(B)])
    dets = [d for d in ("PersonaJB", "PersonaJB_filtered", "raw") if d in V] + [f"Gaussian{g}" for g in range(G)]
    T = {}
    for det in dets:
        for a in SETS:
            fams = [f for f in FAM if f != a]; pts, bss = zip(*[boot(V["y"][f], V[det][(a, f)], bidx[f]) for f in fams])
            T[(det, a)] = (np.mean(pts), np.nanmean(np.stack(bss), 0)); res["per_family"][f"{m}__{det}__{a}"] = dict(zip(fams, map(float, pts)))
        T[(det, "avg")] = (np.mean([T[(det, a)][0] for a in SETS]), np.mean([T[(det, a)][1] for a in SETS], 0))
    for a in SETS + ["avg"]:
        T[("Gaussian", a)] = (np.mean([T[(f"Gaussian{g}", a)][0] for g in range(G)]), np.mean([T[(f"Gaussian{g}", a)][1] for g in range(G)], 0))
        c = {}
        for det in [d for d in ("PersonaJB", "PersonaJB_filtered", "raw") if d in V] + ["Gaussian"]:
            c[det] = float(T[(det, a)][0]); c[f"{det}_ci95"] = ci(T[(det, a)][1]); SC[f"boot__{m}__{det}__{a}"] = T[(det, a)][1].astype(np.float32)
        for p in [d for d in ("PersonaJB", "PersonaJB_filtered") if d in V]:
            for o in ("Gaussian", "raw"):
                c[f"{p}-{o}"] = c[p] - c[o]; c[f"{p}-{o}_ci_bonf"] = dci(T[(p, a)][1] - T[(o, a)][1])
        res["cells"][f"{m}__{a}"] = c
json.dump(res, open("persona_four.json", "w"), indent=1); np.savez("scores_persona_four.npz", **SC)
for m in M:
    for det in [d for d in ("PersonaJB", "PersonaJB_filtered", "Gaussian", "raw") if d in res["cells"][f"{m}__avg"]]:
        print(f"{m:5s} {det:18s} " + " ".join(f"{a}:{res['cells'][f'{m}__{a}'][det]:.3f}" for a in SETS + ["avg"]))
    for p in [d for d in ("PersonaJB", "PersonaJB_filtered") if d in res["cells"][f"{m}__avg"]]:
        for o in ("Gaussian", "raw"):
            k = f"{p}-{o}"; print(f"{m:5s} {k:18s} " + " ".join(f"{a}:{res['cells'][f'{m}__{a}'][k]:+.3f}[{res['cells'][f'{m}__{a}'][k + '_ci_bonf'][0]:+.3f},{res['cells'][f'{m}__{a}'][k + '_ci_bonf'][1]:+.3f}]" for a in SETS + ["avg"]))
