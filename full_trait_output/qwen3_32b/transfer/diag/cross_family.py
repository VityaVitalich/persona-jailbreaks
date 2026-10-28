"""Cross-family matrix: train a probe on one set, test (folded AUC) on every other; diagonal = 5-fold CV.
Qwen3-32B L48 last token vs Llama-3.1-8B L16 (shared data). Full probe (LR C=0.01) and PersonaJB (240/229 traits, LR C=10).
Saves per-prompt scores to scores_cross_family.npz, matrices to cross_family.json."""
import json, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score; from sklearn.model_selection import StratifiedKFold, cross_val_predict
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"; TR = f"{Q}/transfer"; LD = f"{Q}/llama_control/data"
SETS = ["hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"]
def qwen(st):
    own = st not in ("hb", "wjb", "pap")
    A = torch.load(f"{TR}/acts_pos_{st}.pt" if own else f"{Q}/l48_run/acts_pos_{st}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"{TR}/classified_{st}.jsonl" if own else f"{Q}/stage2_out/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    return A["X"][k, A["layers"].index(48), -1].float().numpy().astype(np.float64), np.array([int(C[A["pair_ids"][i]]["jailbroken"]) for i in k])
def llama(st):
    d = {"hb": "harmbench", "wjb": "wildjailbreak"}.get(st, st)
    try: R = [json.loads(l) for l in open(f"{LD}/{d}/classified_responses.jsonl")]
    except FileNotFoundError: R = [json.loads(l) for l in open(f"{LD}/{d}/responses.jsonl")]
    A = torch.load(f"{LD}/{d}/activations.pt", map_location="cpu", weights_only=False)
    R = [r for r in R if r.get("jailbroken") is not None and r["pair_id"] in A and (st != "hb" or r["attack_type"] == "human_jailbreak")]
    return np.stack([A[r["pair_id"]]["16"].float().numpy() for r in R]).astype(np.float64), np.array([int(r["jailbroken"]) for r in R])
names = json.load(open(f"{Q}/trait_names_layer32.json"))
TQ = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[48].float().numpy() for n in names]).astype(np.float64)
TL = np.load(f"{LD}/trait_matrix_layer16.npy").astype(np.float64)
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
res, sc = {}, {}
for model, loader, T in (("qwen", qwen, TQ), ("llama", llama, TL)):
    T = T / np.linalg.norm(T, axis=1, keepdims=True); D = {s: loader(s) for s in SETS}
    for s in SETS: sc[f"{model}__labels_{s}"] = D[s][1]
    for det, C, P in (("fullprobe", 0.01, lambda X: X), ("PersonaJB", 10.0, lambda X: X @ T.T)):
        M = np.zeros((len(SETS), len(SETS)))
        for i, a in enumerate(SETS):
            Xa, ya = P(D[a][0]), D[a][1]; mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
            oof = cross_val_predict(mk(), Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function"); M[i, i] = fold(ya, oof); sc[f"{model}__{det}__{a}__self_oof"] = oof.astype(np.float32)
            m = mk().fit(Xa, ya)
            for j, b in enumerate(SETS):
                if b == a: continue
                s = m.decision_function(P(D[b][0])); M[i, j] = fold(D[b][1], s); sc[f"{model}__{det}__train_{a}__test_{b}"] = s.astype(np.float32)
        res[f"{model}__{det}"] = M.tolist()
        print(f"\n{model} {det}: rows = train set, cols = test set (diag = 5-fold CV)\n         " + " ".join(f"{s:>8s}" for s in SETS))
        for i, a in enumerate(SETS): print(f"{a:8s} " + " ".join(f"{M[i, j]:8.3f}" for j in range(len(SETS))), flush=True)
np.savez("scores_cross_family.npz", **sc); json.dump({"sets": SETS, **res}, open("cross_family.json", "w"), indent=1)
