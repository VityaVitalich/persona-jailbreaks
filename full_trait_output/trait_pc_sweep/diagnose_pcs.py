#!/usr/bin/env python3
"""Why top trait-space PCs are weak detectors. Per model:
 (1) PCA sanity: our SVD basis vs sklearn PCA on the trait matrix; features = h @ u_i.
 (2) per-PC single-feature AUC on HarmBench (all rows, signed and |.|).
 (3) cumulative share of the jailbreak contrast that the top-k trait PCs capture:
     dmu = mean(h|jb) - mean(h|refused) projected into span(T), and the PersonaJB normal w (fit on seed-0 train pool).
 (4) cumulative share of HarmBench activation variance (inside span(T)) along the top-k PCs, vs the trait-variance share."""
import json, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "."); import trait_pc_sweep as S
res = {}
for m in ["llama", "gemma"]:
    cfg = S.CFG[m]; L = cfg["layer"]
    T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    Tc = T - T.mean(0); _, Sc, Vc = np.linalg.svd(Tc, full_matrices=False); r = int((Sc > 1e-8*Sc[0]).sum()); Vc = Vc[:r]
    sk = PCA().fit(T)
    cos = np.abs(np.sum(sk.components_[:10] * Vc[:10], 1))
    rows = [x for x in S.load_jsonl(cfg["human"][0]) if x.get("attack_type") == "human_jailbreak"]
    X, y, valid = S.build_matrix(rows, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), L); X = X.astype(np.float64)
    # (2) per-PC AUC
    Z = X @ Vc.T
    auc = np.array([roc_auc_score(y, Z[:, i]) for i in range(r)]); sym = np.maximum(auc, 1-auc)
    # (3) jailbreak contrast inside span(T)
    Q, _ = np.linalg.qr(T.T)                         # orthonormal basis of span(T), d x n
    dmu = X[y == 1].mean(0) - X[y == 0].mean(0)
    dmu_in = Q @ (Q.T @ dmu)
    # PersonaJB normal on seed-0 train pool (published recipe: StandardScaler + LR C=10 on Th; w = T^T beta/sigma)
    tr, _ = S.pool_split(valid, 0); P = X[tr] @ T.T
    mdl = S.model(0).fit(P, y[tr]); w = T.T @ (mdl[-1].coef_[0] / mdl[0].scale_)
    # basis with mean dir first so cumulative curves reach 1 at full span
    mu = T.mean(0); mu_c = mu - Vc.T @ (Vc @ mu); mu_c /= np.linalg.norm(mu_c)
    B = np.vstack([Vc, mu_c[None]])                  # PCs first, mean dir last
    def cum(v): e = (B @ v)**2; return np.cumsum(e) / np.sum(e)
    # (4) activation variance inside span(T) along PCs
    Xc = X - X.mean(0); va = np.var(Xc @ B.T, axis=0); act_cum = np.cumsum(va) / va.sum()
    trait_cum = np.cumsum(Sc[:r]**2) / np.sum(Sc[:r]**2)
    # Cohen's d per PC: separation relative to spread
    d = (Z[y==1].mean(0) - Z[y==0].mean(0)) / Z.std(0)
    res[m] = dict(r=r, sk_cos_top10=cos.round(4).tolist(), sk_evr_match=float(np.max(np.abs(sk.explained_variance_ratio_[:r] - Sc[:r]**2/np.sum(Sc**2)))),
                  auc=auc.tolist(), d=d.tolist(), dmu_cum=cum(dmu_in).tolist(), w_cum=cum(w).tolist(), act_cum=act_cum.tolist(), trait_cum=trait_cum.tolist(),
                  dmu_in_share=float(np.sum(dmu_in**2)/np.sum(dmu**2)), mean_dir_share_dmu=float((mu_c@dmu_in)**2/np.sum(dmu_in**2)), mean_dir_share_w=float((mu_c@w)**2/np.sum(w**2)))
    ks = [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]
    print(f"\n=== {m}: rank {r}; sklearn PCA vs ours |cos| top10 = {cos.min():.4f}..{cos.max():.4f}, EVR max diff {res[m]['sk_evr_match']:.1e}")
    print(f"  dmu share inside trait span: {res[m]['dmu_in_share']:.3f}; mean-trait dir share of in-span dmu {res[m]['mean_dir_share_dmu']:.3f}, of w {res[m]['mean_dir_share_w']:.3f}")
    print("  top-k PCs capture:      k=" + " ".join(f"{k:>5d}" for k in ks))
    for name, c in [("trait variance", trait_cum), ("HB act. variance", act_cum), ("in-span dmu energy", res[m]["dmu_cum"]), ("PersonaJB w energy", res[m]["w_cum"])]:
        print(f"  {name:22s}  " + " ".join(f"{c[k-1]:5.2f}" for k in ks))
    top = np.argsort(-sym)[:10]
    print("  per-PC AUC, PC1..10:   " + " ".join(f"{a:.2f}" for a in sym[:10]))
    print("  best single PCs (index:AUC): " + " ".join(f"{i+1}:{sym[i]:.2f}" for i in top))
    print(f"  #PCs with |AUC-.5|>.1: {(np.abs(auc-.5)>.1).sum()} ; their indices median {np.median(np.where(np.abs(auc-.5)>.1)[0]+1):.0f}")
json.dump(res, open("diagnose_pcs.json", "w"))
