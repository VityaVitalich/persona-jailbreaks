#!/usr/bin/env python3
"""Per-token-position probe, RECAST to the main detector's task and protocol.

Same label and split as the deployed trait detector (jailbroken vs refused; the deployed
model is StandardScaler + LogisticRegression(C=10, l2) on the 229-trait projections, layer
16, HarmBench CV AUC 0.880). At each token position relative to the pre-generation readout
token we fit a FRESH probe with that exact recipe, using the main behavior+template-disjoint
pool split averaged over 50 seeds. Position 0 (the readout token) therefore reproduces the
detector's in-distribution HarmBench AUC, and earlier positions show where that same
jailbroken-vs-refused signal emerges. All 20 human-jailbreak templates (not a subset).

GPU collection + offline probe fitting."""
import json, pickle, random
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PKL = "full_trait_output/paper_data/detection_llama/trained_models/trait_detector.pkl"
HB = "full_trait_output/paper_data/activations/harmbench"
HB_LOCAL = "../HarmBench/baselines/human_jailbreaks/jailbreaks.py"  # sibling of repo, works under /mnt too
L = 16
W = 26
N_SEEDS = 50
TRAIN_FRAC = 0.7
OUT = Path("full_trait_output"); OUT.mkdir(parents=True, exist_ok=True)

T = pickle.load(open(PKL, "rb"))["trait_matrix"]              # (229, 4096)
rows = [json.loads(l) for l in open(f"{HB}/classified_responses.jsonl") if l.strip()]

# all human-jailbreak pairs, all 20 templates; label = jailbroken (complied) vs refused
prompts = []   # (behavior_id, jailbreak_idx, behavior_text, y)
for r in rows:
    if r.get("attack_type") != "human_jailbreak" or r.get("jailbroken") is None:
        continue
    prompts.append((r["behavior_id"], r["jailbreak_idx"], r["behavior_text"], int(bool(r["jailbroken"]))))
n_tpl = len({p[1] for p in prompts}); n_beh = len({p[0] for p in prompts})
print(f"prompts={len(prompts)} templates={n_tpl} behaviors={n_beh} "
      f"jb_rate={np.mean([p[3] for p in prompts]):.3f}", flush=True)

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def fix_surrogates(s):
    """Some jailbreak templates embed emoji as lone UTF-16 surrogate pairs, which the fast
    tokenizer rejects. Round-trip through UTF-16 to reconstitute the real emoji."""
    return s.encode("utf-16", "surrogatepass").decode("utf-16")


def tail_proj(text):
    """229-trait projection at each of the last W+1 token positions, right-aligned
    (index W = readout token); NaN where the prompt is shorter than the window."""
    chat = tok.apply_chat_template([{"role": "user", "content": fix_surrogates(text)}], tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    proj = hs @ T.T                                  # [n_tok, 229]
    seg = np.full((W + 1, 229), np.nan, dtype=np.float32)
    t = proj[-(W + 1):]
    seg[W + 1 - len(t):] = t
    return seg


X, Y, BEH, TPL = [], [], [], []
fails = []
for i, (bid, tmpl, text, y) in enumerate(prompts):
    try:
        seg = tail_proj(JB[tmpl].format(text))
    except Exception as e:
        fails.append((i, tmpl, f"{type(e).__name__}: {e}"))
        if len(fails) <= 3:
            print(f"  FAIL i={i} tmpl={tmpl}: {type(e).__name__}: {e}", flush=True)
            print("    content[:160]:", repr(JB[tmpl].format(text)[:160]), flush=True)
        continue
    X.append(seg); Y.append(y); BEH.append(bid); TPL.append(tmpl)
    if i % 200 == 0:
        print(f"  {i}/{len(prompts)}  (kept={len(X)} failed={len(fails)})", flush=True)
X = np.stack(X); Y = np.array(Y); BEH = np.array(BEH); TPL = np.array(TPL)
from collections import Counter
print(f"collected kept={len(X)} failed={len(fails)}", flush=True)
if fails:
    print("failed-by-template:", dict(Counter(t for _, t, _ in fails)), flush=True)
np.savez(OUT / "per_position_probe_recast_data.npz", X=X, Y=Y, BEH=BEH, TPL=TPL, W=W)
print("collected", X.shape, flush=True)


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    a = float(roc_auc_score(y_true, score))
    return max(a, 1.0 - a)


def pool_split(seed):
    """Replicates run_all_traits_sweep_v2.get_pool_split: behavior AND template disjoint."""
    rng = random.Random(seed)
    behaviors = sorted(set(BEH.tolist())); templates = sorted(set(TPL.tolist()))
    rng.shuffle(behaviors); rng.shuffle(templates)
    nb = max(1, int(len(behaviors) * TRAIN_FRAC)); nt = max(1, int(len(templates) * TRAIN_FRAC))
    tr_b, te_b = set(behaviors[:nb]), set(behaviors[nb:])
    tr_t, te_t = set(templates[:nt]), set(templates[nt:])
    train = np.array([b in tr_b and t in tr_t for b, t in zip(BEH, TPL)])
    test = np.array([b in te_b and t in te_t for b, t in zip(BEH, TPL)])
    return train, test


# per-position probe: main recipe (logreg_l2 C=10), 50-seed behavior+template-disjoint mean
pos_mean, pos_std, pos_n = [], [], []
for p in range(W + 1):
    Xp = X[:, p, :]
    ok = ~np.isnan(Xp).any(axis=1)
    seed_aucs, ntest = [], []
    for seed in range(N_SEEDS):
        train, test = pool_split(seed)
        trp = train & ok; tep = test & ok
        if trp.sum() < 20 or tep.sum() < 20 or len(set(Y[trp])) < 2 or len(set(Y[tep])) < 2:
            continue
        clf = Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                                                   max_iter=4000, random_state=seed))]).fit(Xp[trp], Y[trp])
        seed_aucs.append(safe_auc(Y[tep], clf.decision_function(Xp[tep]))); ntest.append(int(tep.sum()))
    pos_mean.append(float(np.nanmean(seed_aucs)) if seed_aucs else np.nan)
    pos_std.append(float(np.nanstd(seed_aucs)) if seed_aucs else np.nan)
    pos_n.append(int(np.mean(ntest)) if ntest else 0)
pos_mean = np.array(pos_mean); pos_std = np.array(pos_std)
np.savez(OUT / "per_position_probe_recast_auc.npz", aucs=pos_mean, stds=pos_std, ns_te=np.array(pos_n), W=W)
x = np.arange(-W, 1)
for p, a, s, n in zip(x, pos_mean, pos_std, pos_n):
    print(f"pos {p:+3d}  AUC={a:.3f} +/- {s:.3f}  n_test~{n}", flush=True)
print("DONE", flush=True)
