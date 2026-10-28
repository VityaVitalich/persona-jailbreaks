#!/usr/bin/env python3
"""Is the Gemma PersonaJB steering vector sound? For the steering bundles' vectors (Gemma layer 30, Llama layer 16):
  AUC        folded HB AUC of h.v (all human-jailbreak rows) - for w_persona must equal the detector's ranking
  refit cos  cos(bundle w_persona, normal of a fresh refit: scaler + LR on projections onto the trait vectors)
  std        std of h.v over HB (+WJB) prompts, relative to the std along a random unit direction (median of 200)
             = how far the data naturally spreads along v; the hook adds alpha * RN * v (RN 258 Gemma / 22 Llama)
  sigma@.2   alpha=0.2 push in units of that std
  massive    share of v's squared norm in the 5 largest-|activation| dims (Gemma: 92% of an activation's norm)
  cos pat    cos(v, Cov v / |Cov v|): weight vector vs its activation pattern (Haufe et al. 2014)"""
import sys, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
O = Path.home() / "persona-vector-jbing/full_trait_output"; sys.path.insert(0, str(O / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402
fa = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)); unit = lambda v: v / np.linalg.norm(v)
def audit(model, vecs, RN, C):
    cfg = S.CFG[model]; L = cfg["layer"]
    hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), L); X = Xh.astype(np.float64); y = yh
    if cfg["aug"]:
        Xa, ya, _ = S.build_matrix(S.load_jsonl(cfg["aug"][0]), torch.load(cfg["aug"][1], map_location="cpu", weights_only=False), L)
        XF, yF = np.vstack([X, Xa]), np.r_[y, ya]
    else: XF, yF = X, y
    T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    S.C = C; m = S.model(0).fit(XF @ T.T, yF); refit = unit(T.T @ (m[-1].coef_[0] / m[0].scale_))
    Xc = XF - XF.mean(0); G = np.random.default_rng(0).standard_normal((200, X.shape[1])); G /= np.linalg.norm(G, axis=1, keepdims=True)
    sd_rand = np.median((Xc @ G.T).std(0)); top5 = np.argsort(-np.abs(XF).mean(0))[:5]
    print(f"\n===== {model} (layer {L}, RN {RN}); random-direction std {sd_rand:.2f}; activation norm median {np.median(np.linalg.norm(XF, axis=1)):.1f}")
    print(f"{'vector':26s} {'AUC':>6s} {'refit cos':>9s} {'std':>7s} {'std/rand':>8s} {'sigma@.2':>8s} {'massive':>7s} {'cos pat':>7s}")
    for n, v in vecs.items():
        v = unit(np.asarray(v, np.float64)); p = Xc @ v; sd = p.std(); pat = unit(Xc.T @ p)
        print(f"{n:26s} {fa(y, X @ v):6.3f} {abs(v @ refit) if 'ersona' in n else float('nan'):9.3f} {sd:7.2f} {sd / sd_rand:8.2f} {0.2 * RN / sd:8.1f} {np.sum(v[top5] ** 2):7.3f} {abs(v @ pat):7.3f}")
gb = torch.load(O / "gemma_steering_inputs/steering_directions_gemma_layer30.pt", weights_only=False)
print("gemma bundle meta w_persona:", {k: v for k, v in gb["meta"]["w_persona__full"].items() if not isinstance(v, (list, dict))})
audit("gemma", {k: gb["vectors"][k].double().numpy() for k in ["w_persona__full", "refusal_pos1_neg__full", "dmu__full", "w_raw__full", "trait__emotional", "trait__stoic", "random_full_s0__full"]}, 258.088, 1.0)
lb = torch.load(O / "norm_share_controls/steer_controls_v2_layer16.pt", weights_only=False)["vectors"]
print("\nllama bundle keys:", list(lb)[:30])
audit("llama", {k: lb[k].double().numpy() for k in lb if k in ("persona", "refusal", "dmu", "random", "alpacajb_d2", "sel_pca_1")}, 22.0, 1.0)
