"""Gemma-4-31B-it counterpart of the preamble of persona_part_test.py (exec'd by the gemma/ ports of the
norm_share_controls analyses; defines the same names). Layer 30, last token; 240 trait vectors (cluster copy);
refusal = -(harmful - harmless) mean difference, layer 30, last token (steering bundle refusal_pos1_neg__full);
dictionaries of K = 240 rows (control_bases_gemma.npz, extra_bases_gemma.npz, build_bases_gemma.py).
Detector = Gemma PersonaJB protocol of trait_pc_sweep.py: StandardScaler + LR C=10 fit on the HarmBench train pool
plus all 2000 WildJailbreak rows; HarmBench test = held-out pool; transfer = mean over the 5 Gemma attack families."""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
GEM = Path.home() / "persona-vector-jbing/full_trait_output/norm_share_controls/gemma"; OUT = GEM.parent.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(GEM.parent))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402
fold = lambda a: np.maximum(a, 1 - a)
unit_rows = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
LAYER = 30
cfg = S.CFG["gemma"]
T = unit_rows(np.load(cfg["T"]).astype(np.float64)); K = T.shape[0]
cb, eb = np.load(GEM / "control_bases_gemma.npz"), np.load(GEM / "extra_bases_gemma.npz")
ref = torch.load(OUT / "gemma_steering_inputs/steering_directions_gemma_layer30.pt", weights_only=False)["vectors"]["refusal_pos1_neg__full"].double().numpy()
ref = -ref / np.linalg.norm(ref)  # Llama convention: ref = Arditi direction (harmful - harmless); scripts use -ref
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, hv = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), LAYER)
Xh = Xh.astype(np.float64)
Xa, ya, _ = S.build_matrix(S.load_jsonl(cfg["aug"][0]), torch.load(cfg["aug"][1], map_location="cpu", weights_only=False), LAYER)
Xa = Xa.astype(np.float64)
fam = {}
for f, (rp, ap) in cfg["fams"].items():
    X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), LAYER)
    fam[f] = (X.astype(np.float64), y)
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(10)]
sv = lambda V: np.median(fold(rank_auc(yh, Xh @ V.T)))

def detector(V):
    hbs, xfs = [], []
    Za = Xa @ V.T
    for tr, te in splits:
        m = S.model(0).fit(np.vstack([Xh[tr] @ V.T, Za]), np.concatenate([yh[tr], ya]))
        hbs.append(fold(rank_auc(yh[te], m.decision_function(Xh[te] @ V.T)[:, None]))[0])
        xfs.append(np.mean([fold(rank_auc(y, m.decision_function(X @ V.T)[:, None]))[0] for X, y in fam.values()]))
    return float(np.nanmean(hbs)), float(np.mean(xfs))
SPANS = {"Persona traits": T, "Random Gaussian": cb["gauss_d0"].astype(np.float64),
         "HarmBench PCA-240": cb["hb_pca"].astype(np.float64), "Alpaca PCA-240": cb["alpaca_pca"].astype(np.float64)}
DICTS = {"Wikipedia": eb["wiki_d0"], "No Robots": eb["nr_d0"], "Alpaca": cb["acts_d0"]}
