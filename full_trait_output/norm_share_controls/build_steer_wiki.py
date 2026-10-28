#!/usr/bin/env python3
"""
Extra-dictionary steering arms: same construction as build_steer_controls.py with two new non-persona
dictionaries of 229 random unit activations (L16, last token, official Llama-3.1-8B-Instruct):

  wiki  WikiText-103 paragraphs read as plain text, no chat template (wiki/wiki_acts.pt "raw"),
        so the dictionary is not a set of requests and does not sit on the harmful-vs-harmless request axis
  nr    No Robots (the human-written benign component of the Tulu-3 SFT mixture), first user turn,
        Chat category dropped, read as a user turn with the chat template (wiki/norobots_acts.pt "chat"),
        i.e. benign instructions that are not Alpaca

Per dictionary:
  {src}jb_d0..2   PersonaJB detector (StandardScaler + LR C=10) fit on all 3339 labelled HarmBench rows on the
                  projections onto 229 random unit activations (3 draws), normal mapped back, unit
  sel_{src}_1..3  the 3 directions of draw 0 with single-direction AUC farthest from 0.5, oriented to jb
Writes extra_bases_llama.npz ({src}_d0..4) and steer_extra_layer16.pt (steer_fixed_direction.py format).
Usage: python3.13 build_steer_wiki.py [wiki] [nr]   (sources whose activation file exists)
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
from llama_50seed_span import load_direction  # noqa: E402

K = 229
SRC = {"wiki": ("wiki/wiki_acts.pt", "raw", "Wikipedia passages, plain text"),
       "nr": ("wiki/norobots_acts.pt", "chat", "No Robots benign instructions, user turn")}
unit = lambda v: v / np.linalg.norm(v)
unit_rows = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)

ref = load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt")
T = np.load(S.CFG["llama"]["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
A = np.load(HERE / "control_bases_llama.npz")["acts_d0"].astype(np.float64)
cfg = S.CFG["llama"]
X, y, _ = S.build_matrix(S.load_jsonl(cfg["human"][0]), torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
X = X.astype(np.float64)
P = torch.load(HERE / "steer_controls_v2_layer16.pt", weights_only=False)["vectors"]["persona"].double().numpy()

bases, vec, meta = {}, {}, {}
def add(n, v, desc, **kw):
    v = unit(v); vec[n] = torch.tensor(v, dtype=torch.float32)
    meta[n] = {"description": desc, "hb_auc_toward_jb": float(roc_auc_score(y, X @ v)),
               "cos_persona": float(v @ P), "cos_refusal_neg": float(-(v @ ref)), **kw}

print("median |cos| of dictionary rows with the refusal direction:")
for n, V in [("traits", T), ("alpaca acts", A)]:
    print(f"  {n:22s} {np.median(np.abs(V @ ref)):.3f}  (max {np.abs(V @ ref).max():.3f})")
for src in sys.argv[1:] or list(SRC):
    path, reading, label = SRC[src]
    D = torch.load(HERE / path, map_location="cpu")
    for rd in D:  # diagnostic for both readings
        M = unit_rows(np.stack([v.float().numpy() for v in D[rd].values()]).astype(np.float64))
        print(f"  {src + ' ' + rd:22s} {np.median(np.abs(M @ ref)):.3f}  (max {np.abs(M @ ref).max():.3f})")
    ids = sorted(D[reading], key=lambda s: int(s.split("_")[1]))
    M = np.stack([D[reading][i].float().numpy() for i in ids]).astype(np.float64)
    rng = np.random.default_rng(1)
    picks = [rng.choice(len(M), K, replace=False) for _ in range(5)]
    for d, p in enumerate(picks): bases[f"{src}_d{d}"] = unit_rows(M[p])
    for d in range(3):
        V = bases[f"{src}_d{d}"]; m = S.model(0).fit(X @ V.T, y)
        add(f"{src}jb_d{d}", V.T @ (m[-1].coef_[0] / m[0].scale_), f"same detector on 229 random {label}, draw {d}")
    V = bases[f"{src}_d0"]; a = np.array([roc_auc_score(y, X @ v) for v in V])
    for r, i in enumerate(np.argsort(-np.abs(a - 0.5))[:3], 1):
        sgn = 1.0 if a[i] >= 0.5 else -1.0
        add(f"sel_{src}_{r}", sgn * V[i], f"rank-{r} single {src}_d0 direction by |AUC-0.5|, oriented to jb",
            source_id=ids[picks[0][i]], sign=sgn)
np.savez(HERE / "extra_bases_llama.npz", **{k: v.astype(np.float32) for k, v in bases.items()})
torch.save({"vectors": vec, "meta": meta, "layer": 16, "note": "wiki / No Robots steering arms, build_steer_wiki.py"},
           HERE / "steer_extra_layer16.pt")
print(f"\n{'arm':12s} {'HB AUC':>7s} {'cos PJB':>8s} {'cos -ref':>8s}  source")
for n, m in meta.items():
    print(f"{n:12s} {m['hb_auc_toward_jb']:7.3f} {m['cos_persona']:+8.3f} {m['cos_refusal_neg']:+8.3f}  {m.get('source_id', '')}")
