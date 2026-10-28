#!/usr/bin/env python3
"""
Do ordinary activations predict jailbreaks BECAUSE of their component inside persona space?
Each dictionary vector v (Wikipedia / No Robots / Alpaca activation) is split into v_in = P v (inside a 229-dim
span) and v_out = (I - P) v. Span = persona traits, or controls: random Gaussian 229, HarmBench PCA-229 (top
variance of the jailbreak prompts, no labels), Alpaca PCA-229.
  single vector: folded HarmBench AUC of h.v, h.v_in, h.v_out, h.(v_in minus refusal component); share ||v_in||^2
  detector: PersonaJB recipe (StandardScaler + LR C=10) on 229 projections onto the v_out rows (does any signal
            live outside the span?), 10 pool splits, HB test + mean of 5 transfer families, folded.
Writes persona_part_test.json.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402
from llama_50seed_span import load_direction  # noqa: E402
fold = lambda a: np.maximum(a, 1 - a)
unit_rows = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
cfg = S.CFG["llama"]
T = unit_rows(np.load(cfg["T"]).astype(np.float64))
cb, eb = np.load(HERE / "control_bases_llama.npz"), np.load(HERE / "extra_bases_llama.npz")
ref = load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt").astype(np.float64)
ref /= np.linalg.norm(ref)
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, hv = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
Xh = Xh.astype(np.float64)
fam = {}
for f, (rp, ap) in cfg["fams"].items():
    X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)
    fam[f] = (X.astype(np.float64), y)
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(10)]
SPANS = {"Persona traits": T, "Random Gaussian": cb["gauss_d0"].astype(np.float64),
         "HarmBench PCA-229": cb["hb_pca"].astype(np.float64), "Alpaca PCA-229": cb["alpaca_pca"].astype(np.float64)}
DICTS = {"Wikipedia": eb["wiki_d0"], "No Robots": eb["nr_d0"], "Alpaca": cb["acts_d0"]}
sv = lambda V: np.median(fold(rank_auc(yh, Xh @ V.T)))

def detector(V):
    hbs, xfs = [], []
    for tr, te in splits:
        m = S.model(0).fit(Xh[tr] @ V.T, yh[tr])
        hbs.append(fold(rank_auc(yh[te], m.decision_function(Xh[te] @ V.T)[:, None]))[0])
        xfs.append(np.mean([fold(rank_auc(y, m.decision_function(X @ V.T)[:, None]))[0] for X, y in fam.values()]))
    return float(np.mean(hbs)), float(np.mean(xfs))

res = {}
for dn, V in DICTS.items():
    V = unit_rows(V.astype(np.float64)); full = detector(V)
    print(f"\n=== {dn} activations (229)   single-vector median AUC {sv(V):.3f}; detector HB {full[0]:.3f} transfer {full[1]:.3f}")
    print(f"{'span':20s} {'share in':>8s} | single v_in  v_in-ref  v_out | detector on v_out: HB  transfer")
    for sn, B in SPANS.items():
        Q, _ = np.linalg.qr(B.T); Vin = V @ Q @ Q.T; Vout = V - Vin
        sh = float(np.median(np.sum(Vin ** 2, 1)))
        Vin_nr = Vin - np.outer(Vin @ ref, ref)
        a_in, a_innr, a_out = sv(unit_rows(Vin)), sv(unit_rows(Vin_nr)), sv(unit_rows(Vout))
        d_out = detector(unit_rows(Vout))
        res[f"{dn}|{sn}"] = dict(share=sh, single_in=a_in, single_in_noref=a_innr, single_out=a_out, det_out_hb=d_out[0], det_out_xf=d_out[1])
        print(f"{sn:20s} {sh:8.3f} | {a_in:11.3f} {a_innr:9.3f} {a_out:6.3f} | {d_out[0]:21.3f} {d_out[1]:9.3f}")
json.dump(res, open(HERE / "persona_part_test.json", "w"), indent=1)
