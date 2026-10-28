"""Null baselines for the cosine-similarity claims of the persona-jailbreaks paper
(reviewer cyaF). Reads saved activations / trait matrices / the saved PersonaJB model only.

Claims tested
  (a) tab:pca-personas, tab:pca-poles(-gemma): top trait cosines of each PC pole.
      PCA recipe = pca_direction_decomposition.py (raw, unscaled HB human_jailbreak
      activations, 4 comps, random_state 42).
  (b) fig:jailbreak-pca: predictive traits align with the PC1-PC2 plane.
  (c) fig:w-trait-align / tab:traits (Llama): trait cosines with the mapped PersonaJB
      normal w_trait = unit(T^T coef_proj) from all_traits_sweep_v2/best_model.pkl.

Nulls
  iso      : isotropic random unit direction in R^d.
  act      : activation-matched random direction unit(Xc^T g), g~N(0,I_n)
             (Gaussian with the empirical activation covariance).
  persona  : random persona-space direction unit(T^T c), c~N(0,I_n_traits).
  perm     : PersonaJB refit (same pipeline, C=10) on label-permuted HB, mapped back.
  nonpred  : the same cosine statistic computed over the least predictive traits.
"""
import json, pickle, warnings
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
RNG = np.random.default_rng(0)
N_NULL = 2000
N_PERM = 200
OUT = "full_trait_output/cosine_nulls"

CFG = {
    "llama": dict(base="full_trait_output/harmbench_activations", layer="16",
                  T="full_trait_output/trait_matrix_layer16.npy",
                  names="full_trait_output/trait_names_layer16.json"),
    "gemma": dict(base="full_trait_output/harmbench_activations_gemma", layer="30",
                  T="full_trait_output/trait_pc_sweep/_gemma_ref/trait_matrix_layer30.npy",
                  names="full_trait_output/trait_pc_sweep/_gemma_ref/trait_names_layer30.json"),
}


def unit_rows(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def load(model):
    c = CFG[model]
    rows = [json.loads(l) for l in open(f"{c['base']}/classified_responses.jsonl") if l.strip()]
    rows = [r for r in rows if r["attack_type"] == "human_jailbreak"]
    acts = torch.load(f"{c['base']}/activations.pt", map_location="cpu", weights_only=False)
    rows = [r for r in rows if r["pair_id"] in acts]
    X = np.stack([acts[r["pair_id"]][c["layer"]].double().numpy() for r in rows])
    y = np.array([int(r["jailbroken"]) for r in rows])
    T = unit_rows(np.load(c["T"]).astype(np.float64))
    names = json.load(open(c["names"]))
    return X, y, T, names


def null_max_cos(T, U):
    """For each null direction (rows of U), max cosine with any trait (one pole)."""
    C = T @ unit_rows(U).T          # [n_traits, n_null]
    return C.max(0), np.sort(C, 0)[::-1][:3].mean(0)


def pct(a, q=95):
    return float(np.percentile(a, q))


def main():
    res = {}
    for model in ["llama", "gemma"]:
        X, y, T, names = load(model)
        n, d = X.shape
        Xc = X - X.mean(0)
        pca = PCA(n_components=4, random_state=42).fit(X)
        comps = pca.components_
        cos = T @ comps.T                                   # [traits, 4]

        # single-trait full-data AUC (folded), the predictive / non-predictive split
        proj = X @ T.T
        auc = np.array([roc_auc_score(y, proj[:, i]) for i in range(len(T))])
        fauc = np.maximum(auc, 1 - auc)
        order = np.argsort(-fauc)
        top5 = order[:5]
        q = len(T) // 4
        nonpred = order[-q:]                                 # least predictive quartile

        # nulls for a pole's top cosine
        iso = RNG.standard_normal((N_NULL, d))
        act = RNG.standard_normal((N_NULL, n)) @ Xc
        iso_top1, iso_top3 = null_max_cos(T, iso)
        act_top1, act_top3 = null_max_cos(T, act)

        pcs = []
        for k in range(4):
            c = cos[:, k]
            pos = np.sort(c)[::-1][:3]; neg = np.sort(-c)[::-1][:3]
            pcs.append(dict(
                pc=k + 1, var_pct=float(100 * pca.explained_variance_ratio_[k]),
                pos_top3=[(names[i], float(c[i])) for i in np.argsort(-c)[:3]],
                neg_top3=[(names[i], float(c[i])) for i in np.argsort(c)[:3]],
                pos_top1=float(pos[0]), neg_top1=float(neg[0]),
                pos_top3_mean=float(pos.mean()), neg_top3_mean=float(neg.mean()),
                abs_cos_median_all=float(np.median(np.abs(c))),
                abs_cos_mean_top5_pred=float(np.abs(c[top5]).mean()),
                abs_cos_mean_nonpred=float(np.abs(c[nonpred]).mean()),
            ))
        # (b) share of each trait's norm in the PC1-PC2 plane
        plane = np.sqrt(cos[:, 0] ** 2 + cos[:, 1] ** 2)
        act_plane = []
        for _ in range(500):  # same statistic for random 2-planes drawn from the activation distribution
            B, _r = np.linalg.qr((RNG.standard_normal((2, n)) @ Xc).T)
            act_plane.append(np.linalg.norm(T @ B, axis=1).mean())
        res[model] = dict(
            n=int(n), d=int(d), jb_rate=float(y.mean()),
            null_top1=dict(iso_p95=pct(iso_top1), act_p95=pct(act_top1), act_p50=pct(act_top1, 50)),
            null_top3mean=dict(iso_p95=pct(iso_top3), act_p95=pct(act_top3), act_p50=pct(act_top3, 50)),
            pcs=pcs,
            pc12_plane=dict(top5_pred=[(names[i], float(fauc[i]), float(plane[i])) for i in top5],
                            top5_pred_mean=float(plane[top5].mean()),
                            nonpred_mean=float(plane[nonpred].mean()),
                            nonpred_fauc_max=float(fauc[nonpred].max()),
                            all_mean=float(plane.mean()),
                            rank_of_top5=[int(np.where(np.argsort(-plane) == i)[0][0]) + 1 for i in top5],
                            act_random_plane_mean_p95=pct(act_plane),
                            iso_expected=float(np.sqrt(2 / d))),
            top_fauc=[(names[i], float(fauc[i])) for i in order[:5]],
        )

        if model == "llama":
            bm = pickle.load(open("full_trait_output/all_traits_sweep_v2/best_model.pkl", "rb"))
            assert bm["trait_names"] == names
            w = bm["trait_matrix"].astype(np.float64).T @ bm["coef_proj"]
            w /= np.linalg.norm(w)
            cw = T @ w
            absw = np.sort(np.abs(cw))[::-1]
            Z = X @ T.T

            def refit(yy, seed):
                m = Pipeline([("sc", StandardScaler()),
                              ("clf", LogisticRegression(C=10.0, max_iter=4000, random_state=seed))]).fit(Z, yy)
                v = T.T @ (m.named_steps["clf"].coef_[0] / m.named_steps["sc"].scale_)
                return v / np.linalg.norm(v)

            w_refit = refit(y, 42)
            perm_top1, perm_top15 = [], []
            for s in range(N_PERM):
                v = refit(RNG.permutation(y), s)
                a = np.sort(np.abs(T @ v))[::-1]
                perm_top1.append(a[0]); perm_top15.append(a[14])
            pers = unit_rows(RNG.standard_normal((N_NULL, len(T))) @ T)
            pa = np.sort(np.abs(pers @ T.T), 1)[:, ::-1]
            isoa = np.sort(np.abs(unit_rows(iso) @ T.T), 1)[:, ::-1]
            res[model]["w_trait"] = dict(
                cos_best_model_vs_refit=float(w @ w_refit),
                obs_top1=float(absw[0]), obs_15th=float(absw[14]),
                obs_top5=[(names[i], float(cw[i])) for i in np.argsort(-np.abs(cw))[:5]],
                abs_mean_top5_pred=float(np.abs(cw[top5]).mean()),
                abs_mean_nonpred=float(np.abs(cw[nonpred]).mean()),
                abs_median_all=float(np.median(np.abs(cw))),
                iso_top1_p95=pct(isoa[:, 0]), iso_15th_p95=pct(isoa[:, 14]),
                persona_top1_p95=pct(pa[:, 0]), persona_15th_p95=pct(pa[:, 14]),
                persona_top1_p50=pct(pa[:, 0], 50),
                perm_top1_p95=pct(perm_top1), perm_15th_p95=pct(perm_top15),
                perm_top1_p50=pct(perm_top1, 50),
                perm_top1_frac_ge_obs=float(np.mean(np.array(perm_top1) >= absw[0])),
            )
        print(model, json.dumps(res[model], indent=1)[:200])
    json.dump(res, open(f"{OUT}/cosine_nulls.json", "w"), indent=1)


if __name__ == "__main__":
    main()
