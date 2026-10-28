"""Shared OLMo-3-7B probe pieces (data loading, tuned-C LR protocol) for olmo_probe.py and olmo_gaussian.py."""
import json, random, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score; from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
import os
TR = os.environ.get("OLMO_TR", "/Users/viktor/persona-vector-jbing/full_trait_output/olmo3_7b/transfer")  # OLMO_TR: clariden copy
SETS = ["hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"]; FAM = ["gcg", "pair", "pap", "gptfuzz", "pez"]
GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; B = 1000; MAIN = 16
import os
MISSING = [s for s in SETS if not os.path.exists(f"{TR}/classified_{s}.jsonl")]  # preview without a set whose run is pending
SETS = [s for s in SETS if s not in MISSING]; FAM = [f for f in FAM if f not in MISSING]; TAG = "" if not MISSING else "_without_" + "_".join(MISSING)
def load(st):
    A = torch.load(f"{TR}/acts_{st}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"{TR}/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    R = [C[A["pair_ids"][i]] for i in k]
    return A["X"][k, :, -1].float().numpy(), np.array([int(r["jailbroken"]) for r in R]), R, A["layers"]
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
def val_split(R, y, hb, seed0=0):
    for seed in range(seed0, seed0 + 200):
        rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in R}); rng.shuffle(beh); vb = set(beh[:int(0.2 * len(beh))])
        if hb:
            tp = sorted({r["jailbreak_idx"] for r in R}); rng.shuffle(tp); vt = set(tp[:int(0.2 * len(tp))])
            va = np.array([i for i, r in enumerate(R) if r["behavior_id"] in vb and r["jailbreak_idx"] in vt]); tr = np.array([i for i, r in enumerate(R) if r["behavior_id"] not in vb and r["jailbreak_idx"] not in vt])
        else:
            va = np.array([i for i, r in enumerate(R) if r["behavior_id"] in vb]); tr = np.array([i for i, r in enumerate(R) if r["behavior_id"] not in vb])
        if 0 < y[va].sum() < len(va) and 0 < y[tr].sum() < len(tr): return tr, va
    raise RuntimeError("no valid val split")
def tune(F, y, R, hb):
    tr, va = val_split(R, y, hb); vs = {C: mk(C).fit(F[tr], y[tr]).decision_function(F[va]).astype(np.float32) for C in GRID}
    vauc = {C: fold(y[va], s) for C, s in vs.items()}; return max(GRID, key=lambda c: vauc[c]), vauc, va, vs
def cell(li, a, Fa, ya, Ra, tests):
    Cb, vauc, va, vs = tune(Fa, ya, Ra, a == "hb"); m = mk(Cb).fit(Fa, ya)
    return (li, a), {"C": Cb, "val_auc": vauc, "val_idx": va, "val_scores": vs, "test": {b: m.decision_function(Fb).astype(np.float32) for b, Fb in tests.items()}}
def within(a, F, y, R):
    g = np.array([r["behavior_id"] for r in R]); oof = np.zeros(len(y), np.float32); Cs = []
    for tr, te in GroupKFold(5).split(F, y, g):
        Rt = [R[i] for i in tr]; Cb = tune(F[tr], y[tr], Rt, a == "hb")[0]; Cs.append(Cb)
        oof[te] = mk(Cb).fit(F[tr], y[tr]).decision_function(F[te])
    return a, {"oof": oof, "C": Cs}

# per-model loaders at each model's readout (X float64 [n, d], y, rows), used by four_models.py / persona_four.py
GD = "/Users/viktor/persona-vector-jbing/full_trait_output/olmo3_7b/gemma_data"
def gemma(st):
    d = {"hb": "harmbench", "wjb": "wildjailbreak"}.get(st, st)
    R = [json.loads(l) for l in open(f"{GD}/{d}/classified_responses.jsonl")]
    A = torch.load(f"{GD}/{d}/activations.pt", map_location="cpu", weights_only=False)
    R = [r for r in R if r.get("jailbroken") is not None and r["pair_id"] in A and (st != "hb" or r["attack_type"] == "human_jailbreak")]
    return np.stack([A[r["pair_id"]]["30"].float().numpy() for r in R]).astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R
def olmo(st):
    X, y, R, layers = load(st); return X[:, layers.index(16)].astype(np.float64), y, R

# Llama (L16, shared data) and Qwen (L48, last token) loaders: row order identical to qwen3_32b/transfer/diag/cross_family_tuned.py
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"; LD = f"{Q}/llama_control/data"; OD = "/Users/viktor/persona-vector-jbing/full_trait_output/olmo3_7b"
def llama(st):
    d = {"hb": "harmbench", "wjb": "wildjailbreak"}.get(st, st)
    try: R = [json.loads(l) for l in open(f"{LD}/{d}/classified_responses.jsonl")]
    except FileNotFoundError: R = [json.loads(l) for l in open(f"{LD}/{d}/responses.jsonl")]
    A = torch.load(f"{LD}/{d}/activations.pt", map_location="cpu", weights_only=False)
    R = [r for r in R if r.get("jailbroken") is not None and r["pair_id"] in A and (st != "hb" or r["attack_type"] == "human_jailbreak")]
    return np.stack([A[r["pair_id"]]["16"].float().numpy() for r in R]).astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R
def qwen(st):
    own = st not in ("hb", "wjb", "pap")
    A = torch.load(f"{Q}/transfer/acts_pos_{st}.pt" if own else f"{Q}/l48_run/acts_pos_{st}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"{Q}/transfer/classified_{st}.jsonl" if own else f"{Q}/stage2_out/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    R = [C[A["pair_ids"][i]] for i in k]
    return A["X"][k, A["layers"].index(48), -1].float().numpy().astype(np.float64), np.array([int(r["jailbroken"]) for r in R]), R
def trait_matrix(m):
    """Unit trait vectors at the model's readout, rows in sorted-name order (Llama: the paper matrix order)."""
    import glob
    if m == "llama": T = np.load(f"{LD}/trait_matrix_layer16.npy").astype(np.float64); names = json.load(open(f"{LD}/trait_names_layer16.json"))
    elif m == "qwen":
        names = json.load(open(f"{Q}/trait_names_layer32.json")); T = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[48].float().numpy() for n in names]).astype(np.float64)
    else:
        pat, row = {"olmo": (f"{OD}/trait_vectors/all_traits_no_filter/*.pt", 16), "gemma": (f"{OD}/gemma_data/trait_vectors/*.pt", 0)}[m]
        P = sorted(glob.glob(pat)); names = [os.path.basename(p)[:-3] for p in P]
        T = np.stack([torch.load(p, map_location="cpu", weights_only=False)["vector"][row].float().numpy() for p in P]).astype(np.float64)
    return T / np.linalg.norm(T, axis=1, keepdims=True), names
LOADERS = {"llama": llama, "gemma": gemma, "qwen": qwen, "olmo": olmo}

def boot_auc(y, s, Wb):
    """Folded AUC point estimate and its bootstrap draws for weight matrix Wb [B, n] (exact, ties counted 1/2), orientation
    fixed at the point estimate. Draws with no positive or no negative weight are nan."""
    from scipy.sparse import csr_matrix
    u, inv = np.unique(s, return_inverse=True); n = len(s)
    Mp = csr_matrix((y.astype(float), (np.arange(n), inv)), shape=(n, len(u))); Mn = csr_matrix((1.0 - y, (np.arange(n), inv)), shape=(n, len(u)))
    def auc(Wm):
        P = (Mp.T @ Wm.T).T; N = (Mn.T @ Wm.T).T; cn = np.cumsum(N, 1) - N
        num = (P * (cn + 0.5 * N)).sum(1); den = P.sum(1) * N.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"): return np.where(den > 0, num / den, np.nan)
    pt = auc(np.ones((1, n)))[0]; bs = auc(np.asarray(Wb, float))
    return (pt, bs) if pt >= 0.5 else (1 - pt, 1 - bs)
