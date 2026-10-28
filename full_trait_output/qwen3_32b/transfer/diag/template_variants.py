"""Qwen3-32B L48 last token: does template concentration of HB positives (one template 100% ASR, top-3 = 71% of positives)
explain poor transfer? Training variants on the HB human-jailbreak pool (always + optional WJB):
  base       template+behaviour-disjoint split (as before)
  drop100    base, minus templates with ASR >= 0.9 in train
  mixed      base, train only on templates with 0 < ASR < 1
  tplweight  base, sample weights so every training template has equal total weight
  behsplit   behaviour-disjoint split only, all templates in training
Transfer = folded AUC on Qwen's own GCG/PAIR/PEZ/GPTFuzz + PAP. Saves per-prompt scores."""
import json, random, collections, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"; TR = f"{Q}/transfer"; FAM = ["gcg", "pair", "pap", "gptfuzz", "pez"]
def load(st):
    own = st not in ("hb", "wjb", "pap")
    A = torch.load(f"{TR}/acts_pos_{st}.pt" if own else f"{Q}/l48_run/acts_pos_{st}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"{TR}/classified_{st}.jsonl" if own else f"{Q}/stage2_out/classified_{st}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (st != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    return A["X"][k, A["layers"].index(48), -1].float().numpy().astype(np.float64), np.array([int(C[A["pair_ids"][i]]["jailbroken"]) for i in k]), [C[A["pair_ids"][i]] for i in k]
Xh, yh, hv = load("hb"); Xa, ya, _ = load("wjb"); E = {f: load(f)[:2] for f in FAM}
tpl = np.array([r["jailbreak_idx"] for r in hv]); tasr = {t: yh[tpl == t].mean() for t in set(tpl)}
names = json.load(open(f"{Q}/trait_names_layer32.json"))
T = np.stack([torch.load(f"{Q}/hf/qwen-3-32b/trait_vectors/{n}.pt", map_location="cpu")[48].float().numpy() for n in names]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
Gs = [np.random.default_rng(i).standard_normal(T.shape) for i in range(3)]; Gs = [g / np.linalg.norm(g, axis=1, keepdims=True) for g in Gs]
fold = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
def split(seed, mode, f=0.7):
    rng = random.Random(seed); beh = sorted({r["behavior_id"] for r in hv}); tp = sorted(set(tpl)); rng.shuffle(beh); rng.shuffle(tp)
    trb = set(beh[:int(len(beh) * f)]); trt = set(tp[:int(len(tp) * f)]) if mode != "behsplit" else set(tp)
    tr = np.array([i for i, r in enumerate(hv) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt])
    if mode == "drop100": tr = tr[[tasr[tpl[i]] < 0.9 for i in tr]]
    if mode == "mixed": tr = tr[[0 < tasr[tpl[i]] < 1 for i in tr]]
    return tr
MODES = ["base", "drop100", "mixed", "tplweight", "behsplit"]
mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
res, sc = {}, {"labels_HB": yh, "pair_ids_HB": np.array([r["pair_id"] for r in hv]), **{f"labels_{f}": y for f, (_, y) in E.items()}}
for mode in MODES:
    S = [s for s in (split(sd, mode) for sd in range(60)) if 0 < yh[s].sum() < len(s)][:20]
    for nm, V, C in [("PersonaJB", T, 10.0)] + [(f"Gaussian{i}", g, 10.0) for i, g in enumerate(Gs)] + [("fullprobe", None, 0.01)]:
        P = (lambda X: X @ V.T) if V is not None else (lambda X: X); h, a = P(Xh), P(Xa); Ef = {f: P(X) for f, (X, _) in E.items()}
        for wjb in (False, True):
            tag = f"{mode}__{nm}__{'hbwjb' if wjb else 'hb'}"; tra = {f: [] for f in FAM}
            for j, tr in enumerate(S[:(5 if V is None else 20)]):
                w = np.ones(len(tr))
                if mode == "tplweight": cnt = collections.Counter(tpl[tr]); w = np.array([len(tr) / len(cnt) / cnt[tpl[i]] for i in tr])
                Xt, yt, wt = (np.vstack([h[tr], a]), np.r_[yh[tr], ya], np.r_[w, np.ones(len(ya))]) if wjb else (h[tr], yh[tr], w)
                m = mk(C).fit(Xt, yt, logisticregression__sample_weight=wt)
                for f in FAM:
                    sf = m.decision_function(Ef[f]); tra[f].append(fold(E[f][1], sf))
                    if nm in ("PersonaJB", "Gaussian0", "fullprobe"): sc[f"{tag}__{f}__s{j}"] = sf.astype(np.float32)
                if nm in ("PersonaJB", "Gaussian0", "fullprobe"): sc[f"{tag}__HB__s{j}"] = m.decision_function(h).astype(np.float32); sc[f"{mode}__idx_train__s{j}"] = tr
            row = {f: float(np.mean(v)) for f, v in tra.items()}; row["transfer_mean"] = float(np.mean(list(row.values()))); res[tag] = row
    for pool in ("hb", "hbwjb"):
        g = np.mean([res[f"{mode}__Gaussian{i}__{pool}"]["transfer_mean"] for i in range(3)])
        print(f"{mode:9s} {pool:5s} n_train~{int(np.median([len(s) for s in S]))} pos~{int(np.median([yh[s].sum() for s in S]))} | PersonaJB {res[f'{mode}__PersonaJB__{pool}']['transfer_mean']:.3f} Gaussian {g:.3f} full {res[f'{mode}__fullprobe__{pool}']['transfer_mean']:.3f} | PersonaJB per fam " + " ".join(f"{f} {res[f'{mode}__PersonaJB__{pool}'][f]:.3f}" for f in FAM), flush=True)
np.savez("scores_template_variants.npz", **sc); json.dump(res, open("template_variants.json", "w"), indent=1)
