#!/usr/bin/env python3
"""
prep_steering_directions_gemma.py

Gemma-4-31B (layer 30) counterpart of prep_steering_directions.py: builds the fixed
unit-norm directions that jailbreaking_tools/steer_fixed_direction_gemma.py steers
with, in the same bundle schema ({"vectors", "meta", "layer"}).

Directions
  w_persona__full      trait-classifier normal mapped back to hidden space, same recipe
                       as the Llama steer-w (persona) anchor: float32 projections on the
                       trait matrix, StandardScaler, LogisticRegression(C=1, max_iter=2000),
                       unit((coef/scale) @ V). Fit on ALL HarmBench human_jailbreak pairs
                       PLUS the 2000 WildJailbreak prompts, because HarmBench alone has
                       only 63 Gemma jailbreaks (the paper's Gemma detection protocol
                       folds WJB into training for the same reason).
  w_raw__full          raw-activation classifier normal (harmbench_logreg_gemma_hbwjb).
  refusal_pos1_neg__full  -(harmful - harmless mean) at hidden_states[31] == layer 30,
                       last prompt token, from gemma_refusal_dir/out (Arditi recipe,
                       128+128 train prompts).
  dmu__full            unit(mean(jailbroken) - mean(refused)) over HB human_jailbreak +
                       WJB (HB-only variant audited, not steered).
  trait__<name>        the ten most predictive single traits of tab:persona-auc-gemma,
                       each unit row of trait_matrix_layer30 ORIENTED toward the
                       jailbroken class (sign = sign of mean projection gap
                       jailbroken - refused on HB+WJB). The sign is stored in meta.
  random_full_s0__full, random_persona_s0__full   matched controls, recipe of
                       build_control_directions.py (default_rng(0)).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TOP10 = ["romantic", "stoic", "dispassionate", "emotional", "literal",
         "flirty", "patient", "dominant", "reserved", "confident"]

def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / np.linalg.norm(v)

def load_rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def load_acts(rows, acts_path, layer):
    acts = torch.load(acts_path, map_location="cpu", weights_only=False)
    X, y, ids = [], [], []
    for r in rows:
        a = acts.get(r["pair_id"])
        if a is None or r.get("jailbroken") is None:
            continue
        t = a[str(layer)] if isinstance(a, dict) else a
        X.append(t.float().numpy()); y.append(int(bool(r["jailbroken"]))); ids.append(r["pair_id"])
    return np.stack(X).astype(np.float64), np.asarray(y), ids

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--hb_classified", default="full_trait_output/harmbench_activations_gemma/classified_responses.jsonl")
    p.add_argument("--hb_acts", default="full_trait_output/harmbench_activations_gemma/activations.pt")
    p.add_argument("--wjb_classified", default="full_trait_output/wildjailbreak_activations_gemma/classified_responses.jsonl")
    p.add_argument("--wjb_acts", default="full_trait_output/wildjailbreak_activations_gemma/activations.pt")
    p.add_argument("--trait_matrix", default="full_trait_output/gemma_sweep_trait/trait_matrix_layer30.npy")
    p.add_argument("--trait_names", default="full_trait_output/gemma_sweep_trait/trait_names_layer30.json")
    p.add_argument("--w_raw", default="full_trait_output/harmbench_logreg_gemma_hbwjb/hyperplane_normal_layer30.pt")
    p.add_argument("--refusal_dir", default="full_trait_output/gemma_refusal_dir/out")
    p.add_argument("--traits", default=",".join(TOP10))
    p.add_argument("--output_dir", default="full_trait_output/gemma_steering_inputs")
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    hb_rows = [r for r in load_rows(a.hb_classified) if r.get("attack_type") == "human_jailbreak"]
    Xh, yh, _ = load_acts(hb_rows, a.hb_acts, a.layer)
    Xw, yw, _ = load_acts(load_rows(a.wjb_classified), a.wjb_acts, a.layer)
    X = np.vstack([Xh, Xw]); y = np.concatenate([yh, yw])
    print(f"HB human_jailbreak {Xh.shape} jb={yh.sum()} | WJB {Xw.shape} jb={yw.sum()} | pooled jb_rate={y.mean():.3f}", flush=True)

    V = np.asarray(np.load(a.trait_matrix), dtype=np.float64)          # (240, 5376) unit rows
    names = json.load(open(a.trait_names))
    assert V.shape[0] == len(names)
    _, _, Vt = np.linalg.svd(V, full_matrices=False)                  # orthonormal basis of the trait span
    Q = Vt.T                                                           # (5376, 240)
    def span_share(d): pd_ = Q @ (Q.T @ d); return float(pd_ @ pd_ / (d @ d))
    def auc(d, XX=X, yy=y):
        s = XX @ d; v = roc_auc_score(yy, s); return float(v)

    # w_persona: Llama anchor recipe, on HB+WJB
    xt = (X @ V.T).astype(np.float32)
    sc = StandardScaler().fit(xt)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(sc.transform(xt), y)
    coef_scaled = clf.coef_.ravel() / sc.scale_
    w_persona = unit(coef_scaled @ V)
    print(f"w_persona fit: train AUC {roc_auc_score(y, clf.decision_function(sc.transform(xt))):.4f}", flush=True)

    w_raw_obj = torch.load(a.w_raw, map_location="cpu", weights_only=False)
    w_raw = unit(w_raw_obj["vector"] if isinstance(w_raw_obj, dict) else w_raw_obj)

    mh = torch.load(Path(a.refusal_dir) / "means_harmful.pt", map_location="cpu", weights_only=False)
    ml = torch.load(Path(a.refusal_dir) / "means_harmless.pt", map_location="cpu", weights_only=False)
    meta_ref = json.load(open(Path(a.refusal_dir) / "meta.json"))
    k = a.layer + 1                                                    # hidden_states[k] == layer 30 (meta convention)
    r = unit((mh[k, -1] - ml[k, -1]).double().numpy())                 # pos -1 = last of the stored 5 positions
    refusal_neg = -r
    print(f"refusal direction from {meta_ref['convention']!r}; AUC(+r) on HB+WJB = {auc(r):.3f}", flush=True)

    dmu = unit(X[y == 1].mean(0) - X[y == 0].mean(0))
    dmu_hb = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))

    vectors, meta, audit = {}, {}, {}
    def add(key, d, desc, **extra):
        d = unit(d)
        vectors[key] = torch.tensor(d, dtype=torch.float32)
        meta[key] = dict(description=desc, share_in_span=span_share(d),
                         auc_hbwjb=auc(d), auc_hb=auc(d, Xh, yh),
                         mean_gap=float(X[y == 1].mean(0) @ d - X[y == 0].mean(0) @ d),
                         cos_w_persona=float(d @ w_persona), cos_dmu=float(d @ dmu), **extra)

    add("w_persona__full", w_persona, "trait-classifier normal (HB+WJB, scaler+LR C=1) mapped to hidden space")
    add("w_raw__full", w_raw, "raw-activation classifier normal (harmbench_logreg_gemma_hbwjb)")
    add("refusal_pos1_neg__full", refusal_neg, "-(harmful-harmless mean diff), layer 30, last token, Arditi recipe")
    add("dmu__full", dmu, "unsupervised jailbreak direction mean(jb)-mean(refused), HB human_jailbreak + WJB",
        cos_dmu_hb_only=float(dmu @ dmu_hb))

    tr_names = [t.strip() for t in a.traits.split(",") if t.strip()]
    print(f"\n{'trait':16s} {'gap':>8s} {'coef':>8s} {'sign':>5s} {'AUC(signed)':>11s}", flush=True)
    for t in tr_names:
        i = names.index(t); v = V[i]
        gap = float(X[y == 1].mean(0) @ v - X[y == 0].mean(0) @ v)
        sign = 1.0 if gap >= 0 else -1.0
        print(f"{t:16s} {gap:8.3f} {coef_scaled[i]:8.3f} {sign:+5.0f} {auc(sign*v):11.3f}", flush=True)
        add(f"trait__{t}", sign * v, f"single trait direction '{t}' oriented toward the jailbroken class",
            trait=t, sign=sign, lr_coef_scaled=float(coef_scaled[i]))

    rng = np.random.default_rng(0)
    add("random_full_s0__full", rng.standard_normal(V.shape[1]), "isotropic random direction, default_rng(0)")
    add("random_persona_s0__full", Q @ rng.standard_normal(V.shape[0]), "uniform random direction in the 240-trait span, default_rng(0)")

    print(f"\n{'direction':28s} {'span':>6s} {'AUC':>6s} {'AUC_hb':>7s} {'gap':>7s} {'cos_wp':>7s} {'cos_dmu':>8s}", flush=True)
    for kname, m in meta.items():
        print(f"{kname:28s} {m['share_in_span']:6.3f} {m['auc_hbwjb']:6.3f} {m['auc_hb']:7.3f} {m['mean_gap']:7.2f} {m['cos_w_persona']:7.3f} {m['cos_dmu']:8.3f}", flush=True)

    bundle_path = out / f"steering_directions_gemma_layer{a.layer}.pt"
    torch.save({"vectors": vectors, "meta": meta, "layer": a.layer, "model": "google/gemma-4-31B-it",
                "fit_pool": "HB human_jailbreak (all) + WJB 2000", "n_fit": int(len(y))}, bundle_path)
    (out / "steering_directions_gemma_audit.json").write_text(json.dumps(
        {"layer": a.layer, "n_hb": int(len(yh)), "n_wjb": int(len(yw)), "meta": meta,
         "random_span_share_expect": V.shape[0] / V.shape[1]}, indent=2))
    print(f"\nwrote {len(vectors)} vectors -> {bundle_path}", flush=True)

if __name__ == "__main__":
    main()
