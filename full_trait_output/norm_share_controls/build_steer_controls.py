#!/usr/bin/env python3
"""
Steering arms for the matched-control steering experiment (Llama-3.1-8B, L16). Does a persona direction
steer better than a non-persona direction built the same way?

  Supervised (vs PersonaJB): the PersonaJB detector (StandardScaler + LR C=10) refit on all labelled
  HarmBench rows (human_jailbreak + direct_request, n=3339, the rows the deployed trait_detector.pkl saw)
  on each 229-direction dictionary, normal mapped back to activation space w = V^T (coef / scale), unit:
    persona_refit   dictionary = the 229 trait vectors
    alpacajb_d0..2  dictionary = 229 random Alpaca activations (control_bases_llama.npz acts_d0..2)
    noisejb_d0      dictionary = 229 shuffled-label Alpaca contrasts (noise_d0)
    gaussjb_d0      dictionary = 229 random Gaussian directions (gauss_d0)
  Selected (vs single traits): the 3 directions of a dictionary with single-direction AUC farthest
  from 0.5 on the same rows, each oriented toward jailbroken:
    sel_trait_1..3, sel_acts_1..3 (acts_d0), sel_pca_1..3 (alpaca_pca)
Plus anchors copied unchanged from the existing bundles: persona (published steer-w), refusal, dmu,
utilitarian, closure_seeking, edgy, random_full_s0, offdomain_gsm8k.
Writes steer_controls_v2_layer16.pt ({'vectors','meta','layer'}, the steer_fixed_direction.py format).
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402
SCR = Path(sys.argv[1])  # dir holding the cluster bundles ablation_/control_directions_layer16.pt

unit = lambda v: v / np.linalg.norm(v)
cfg = S.CFG["llama"]
names = json.load(open(OUT / "trait_names_layer16.json"))
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
B = {k: v.astype(np.float64) for k, v in np.load(HERE / "control_bases_llama.npz").items()}
rows = S.load_jsonl(cfg["human"][0])
X, y, valid = S.build_matrix(rows, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
X = X.astype(np.float64)
print(f"HB labelled rows {len(y)} (jb {y.mean():.3f}); attack types {sorted({r['attack_type'] for r in valid})}")

vec, meta = {}, {}
def add(n, v, desc, **kw):
    v = unit(np.asarray(v, np.float64)); vec[n] = torch.tensor(v, dtype=torch.float32)
    p = X @ v; meta[n] = {"description": desc, "hb_auc_toward_jb": float(roc_auc_score(y, p)), **kw}

for n, V, desc in [("persona_refit", T, "PersonaJB detector refit on all 3339 HB rows, traits"),
                   ("alpacajb_d0", B["acts_d0"], "same detector on 229 random Alpaca activations, draw 0"),
                   ("alpacajb_d1", B["acts_d1"], "draw 1"), ("alpacajb_d2", B["acts_d2"], "draw 2"),
                   ("noisejb_d0", B["noise_d0"], "same detector on 229 shuffled-label Alpaca contrasts"),
                   ("gaussjb_d0", B["gauss_d0"], "same detector on 229 random Gaussian directions")]:
    m = S.model(0).fit(X @ V.T, y)
    add(n, V.T @ (m[-1].coef_[0] / m[0].scale_), desc)

for tag, V, lab in [("trait", T, names), ("acts", B["acts_d0"], None), ("pca", B["alpaca_pca"], None)]:
    a = np.array([roc_auc_score(y, X @ v) for v in V]); order = np.argsort(-np.abs(a - 0.5))[:3]
    for r, i in enumerate(order, 1):
        sgn = 1.0 if a[i] >= 0.5 else -1.0
        add(f"sel_{tag}_{r}", sgn * V[i], f"rank-{r} single direction of {tag} by |AUC-0.5|, oriented to jb",
            index=int(i), name=(lab[i] if lab else None), sign=sgn)

ab = torch.load(SCR / "ablation_directions_layer16.pt", map_location="cpu", weights_only=False)
ct = torch.load(SCR / "control_directions_layer16.pt", map_location="cpu", weights_only=False)
for n in ["persona", "refusal", "dmu", "utilitarian", "closure_seeking", "edgy", "random_full_s0"]:
    add(n, ab["vectors"][n].double().numpy(), ab["meta"][n].get("description", n))
add("offdomain_gsm8k", ct["vectors"]["offdomain_gsm8k__full"].double().numpy(), "GSM8k correct-vs-incorrect diff-of-means")

P = vec["persona"].double().numpy(); R = vec["refusal"].double().numpy()
for n in vec:
    v = vec[n].double().numpy(); meta[n].update(cos_persona=float(v @ P), cos_refusal_neg=float(v @ R))
torch.save({"vectors": vec, "meta": meta, "layer": 16,
            "note": "matched-control steering arms, built by norm_share_controls/build_steer_controls.py"},
           HERE / "steer_controls_v2_layer16.pt")
print(f"{'arm':16s} {'HB AUC':>7s} {'cos PJB':>8s} {'cos -ref':>8s}  name")
for n, m in meta.items():
    print(f"{n:16s} {m['hb_auc_toward_jb']:7.3f} {m['cos_persona']:+8.3f} {m['cos_refusal_neg']:+8.3f}  {m.get('name') or ''}")
