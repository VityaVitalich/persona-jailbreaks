"""Audit: pair_id uniqueness, response<->label alignment (refusal-keyword agreement, no text printed),
within-family 5-fold CV of the full probe and PersonaJB on L48 last token, and response length vs label."""
import json, collections, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict; from sklearn.metrics import roc_auc_score
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"
names = json.load(open(f"{Q}/trait_names_layer32.json"))
T = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[48].float().numpy() for n in names]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
REF = ("i can't", "i cannot", "i can’t", "i'm sorry", "i’m sorry", "i am sorry", "i won't", "i will not", "i'm unable", "i am unable", "i’m unable", "sorry, but", "as an ai", "i must decline", "i can not")
SAVE = {}
for st in ("hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"):
    own = st not in ("hb", "wjb", "pap")
    A = torch.load(f"acts_pos_{st}.pt" if own else f"{Q}/l48_run/acts_pos_{st}.pt")
    rows = [json.loads(l) for l in open(f"classified_{st}.jsonl" if own else f"{Q}/stage2_out/classified_{st}.jsonl")]
    ids = A["pair_ids"]; cid = [r["pair_id"] for r in rows]
    C = {r["pair_id"]: r for r in rows}
    k = [i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    y = np.array([int(C[ids[i]]["jailbroken"]) for i in k]); X = A["X"][k, A["layers"].index(48), -1].float().numpy().astype(np.float64)
    resp = [C[ids[i]]["response_text"] or "" for i in k]
    refuse = np.array([any(t in s[:300].lower() for t in REF) for s in resp]); ln = np.array([len(s) for s in resp])
    agree = ((refuse == 0) == (y == 1)).mean()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    full = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)), X, y, cv=skf, method="decision_function")
    pj = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=5000)), X @ T.T, y, cv=skf, method="decision_function")
    SAVE.update({f"pair_ids_{st}": np.array([ids[i] for i in k]), f"labels_{st}": y, f"refusal_kw_{st}": refuse.astype(np.int8), f"fullprobe_withinCV_oof_{st}": full.astype(np.float32), f"PersonaJB_withinCV_oof_{st}": pj.astype(np.float32)})
    print(f"{st:8s} n {len(y):5d} pos {y.sum():4d} | dup pair_ids acts {len(ids)-len(set(ids))} cls {len(cid)-len(set(cid))} | "
          f"refusal-kw: P(refuse|jb=1) {refuse[y==1].mean():.2f} P(refuse|jb=0) {refuse[y==0].mean():.2f} agree {agree:.2f} | "
          f"resp len AUC {roc_auc_score(y, ln):.2f} | within-family CV AUC full {roc_auc_score(y, full):.3f} PersonaJB {roc_auc_score(y, pj):.3f}", flush=True)
np.savez("diag/scores_audit_L48_last.npz", **SAVE)
