"""Sanity checks for the subspace-control results (see subspace_tests.py / trait_pc_sweep --tag _subspace_controls)."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S
from llama_50seed_span import load_direction
from build_bases import alpaca_acts
cfg = S.CFG["llama"]
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
B = {k: v.astype(np.float64) for k, v in np.load(HERE / "control_bases_llama.npz").items()}
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
X, y, hv = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16); X = X.astype(np.float64)
A = alpaca_acts(); H = X.shape[1]
ref = load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt")
unit = lambda v: v / np.linalg.norm(v)
dmu = unit(X[y == 1].mean(0) - X[y == 0].mean(0))

print("1. control bases contain no HarmBench rows: max |cos| of any acts_d0 row to any HB activation =",
      round(float(np.max(np.abs((B['acts_d0'] @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T)))), 4))

print("\n2. shuffled training labels, same detector (5 seeds; should be ~0.5)")
for name, V in [("traits", T), ("acts_d0", B["acts_d0"]), ("gauss_d0", B["gauss_d0"])]:
    Z = X @ V.T; a = []
    for seed in range(5):
        tr, te = S.pool_split(hv, seed); yt = np.random.default_rng(seed).permutation(y[tr])
        a.append(roc_auc_score(y[te], S.model(seed).fit(Z[tr], yt).predict_proba(Z[te])[:, 1]))
    print(f"   {name:9s} test AUC {np.mean(a):.3f}  {np.round(a, 3)}")

print("\n3. why a random projection keeps the AUC")
Xc = X - X.mean(0); avg = (Xc ** 2).sum() / len(X) / H          # mean variance per direction
for nm, d in [("refusal", ref), ("dmu", dmu)]:
    vd = np.var(Xc @ d); print(f"   {nm}: variance along d = {vd/avg:.0f}x the average direction")
    for b in ["gauss_d0", "gauss_d3"]:
        Q = np.linalg.qr(B[b].T)[0]; pd = Q @ (Q.T @ d); al = pd @ d; r = pd - al * d
        sig, noi = al ** 2 * vd, np.var(Xc @ r)
        print(f"     {b}: P d = {al:.3f} d + r (|r|^2 {r@r:.3f}); score variance from d / from r = {sig/noi:.1f}"
              f" -> corr(score_proj, score_full) {np.corrcoef(Xc @ pd, Xc @ d)[0,1]:.3f}")

print("\n4. share of centered HB activation variance each subspace holds")
tot = (Xc ** 2).sum()
for nm, V in [("persona", T), ("acts_d0", B["acts_d0"]), ("noise_d0", B["noise_d0"]), ("alpaca_pca", B["alpaca_pca"]), ("gauss_d0", B["gauss_d0"])]:
    Q = np.linalg.qr(V.T)[0]; print(f"   {nm:10s} {((Xc @ Q) ** 2).sum() / tot:.3f}")

print("\n5. how easy is HB: one direction = top-1..4 HB PCs, logistic regression, 5 seeds")
U = np.linalg.svd(Xc, full_matrices=False)[2]
for k in [1, 2, 4]:
    a = []
    for seed in range(5):
        tr, te = S.pool_split(hv, seed); Z = X @ U[:k].T
        a.append(roc_auc_score(y[te], S.model(seed).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]))
    print(f"   k={k}: {np.mean(a):.3f}")
