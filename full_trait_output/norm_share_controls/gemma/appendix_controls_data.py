# GEMMA PORT of norm_share_controls/appendix_controls_data.py (gemma/port.py); model-specific inputs swapped, logic unchanged.
#!/usr/bin/env python3
"""
Data for the appendix control figures (Gemma-4-31B-it, layer 30, last prompt token). Saved activations only.

Directions (unit norm):
  refusal          Arditi refusal direction (~/refusal_direction, layer 30, last token)
  dmu              jailbreak direction, mean(jailbroken) - mean(refused) on HarmBench human_jailbreak
                   (all rows for the share; the train rows of each split for AUCs, as in the paper)
  shuf_refusal     the refusal recipe with labels permuted: harmful plain requests (JBB/AdvBench/StrongREJECT)
                   pooled with plain Alpaca requests (roleplay_check synthetic_v2), random relabelling, mean
                   difference
  shuf_dmu         dmu with the jailbroken labels permuted (within the train rows of each split for AUCs)
  strat_refusal / strat_dmu  the same with a stratified permutation: each pseudo-class gets a random half of every
                   real class, so the class mix is equal and no part of the real direction leaks in (the clean null;
                   a plain permutation leaves the halves with slightly different class mixes)
  shuf_alpaca      mean of a random half of the Alpaca activations minus the other half (noise_d0 rows)
Spaces (240-dim spans): trait vectors, Alpaca PCA-240, 240 Alpaca / No Robots / Wikipedia activations (5 draws),
  240 random Gaussian directions (5 draws).
Outputs (appendix_controls_data.json):
  share[target][space]          share of squared norm (mean, min, max over draws)
  share_persona[direction]      share in the trait span (mean, 5th / 95th percentile over shuffle draws)
  auc[direction][variant]       folded AUC on the HarmBench test set (mean over 50 splits, and its 95% CI half
                                width) and mean folded AUC over the 5 transfer families; variants: full, persona
                                component only, persona component removed, random span removed (5 Gaussian spans)
  For shuffled directions every number is the mean over draws (shuf_refusal 20, shuf_dmu 10, shuf_alpaca 20).
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; OUT = HERE.parent.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(OUT.parent / "full_trait_tools"))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402
sys.path.insert(0, "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma"); from gemma_common import _gemma_arditi  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
unit_rows = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
fold = lambda a: np.maximum(a, 1 - a)
qr = lambda B: np.linalg.qr(np.asarray(B, np.float64).T)[0]
rng = np.random.default_rng(0)

cfg = S.CFG["gemma"]
T = unit_rows(np.load(cfg["T"]).astype(np.float64))
cb, eb = np.load(HERE / "control_bases_gemma.npz"), np.load(HERE / "extra_bases_gemma.npz")
ref = unit(_gemma_arditi().astype(np.float64))
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, hv = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 30)
Xh = Xh.astype(np.float64)
fam = {f: tuple(np.asarray(a, np.float64 if i == 0 else int) for i, a in enumerate(S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 30)[:2]))
       for f, (rp, ap) in cfg["fams"].items()}
splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]
splits = [sp for sp in splits if 0 < yh[sp[1]].sum() < len(sp[1])]  # Gemma: 3 of 50 test pools hold no jailbroken prompt
rows = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
v2 = torch.load(Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/synthetic_v2_acts.pt"), map_location="cpu")
pick = lambda f: np.stack([v2[r["id"]].float().numpy() for r in rows if f(r)]).astype(np.float64)
A = pick(lambda r: r["source"] == "alpaca" and r["frame"] == "plain")
Hm = pick(lambda r: r["harm"] == "harmful" and r["frame"] == "plain")
XR = np.vstack([Hm, A]); yR = np.r_[np.ones(len(Hm), int), np.zeros(len(A), int)]

Qp = qr(T)
SPACES = {"persona": [Qp], "alpaca_pca": [qr(cb["alpaca_pca"])], "alpaca_acts": [qr(cb[f"acts_d{i}"]) for i in range(5)],
          "norobots_acts": [qr(eb[f"nr_d{i}"]) for i in range(5)], "wiki_acts": [qr(eb[f"wiki_d{i}"]) for i in range(5)],
          "random": [qr(cb[f"gauss_d{i}"]) for i in range(5)]}
share = lambda Q, d: float(np.sum((Q.T @ unit(d)) ** 2))
dmu_all = unit(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0))
res = {"n": {"hb": len(yh), "alpaca": len(A), "harmful": len(Hm)}, "share": {}, "share_persona": {}, "auc": {}}
for tn, d in (("refusal", ref), ("dmu", dmu_all)):
    res["share"][tn] = {sn: (lambda v: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))))([share(Q, d) for Q in Qs])
                        for sn, Qs in SPACES.items()}

def perm_dir(X, y):  # plain permutation: the two halves differ in class mix, so the real direction leaks in
    p = rng.permutation(y); return unit(X[p == 1].mean(0) - X[p == 0].mean(0))
def strat_labels(y):  # stratified permutation: each pseudo-class gets half of every real class
    p = np.zeros(len(y), int)
    for c in (0, 1):
        i = rng.permutation(np.where(y == c)[0]); p[i[: len(i) // 2]] = 1
    return p
def strat_dir(X, y):
    p = strat_labels(y); return unit(X[p == 1].mean(0) - X[p == 0].mean(0))
dist = lambda v: dict(mean=float(np.mean(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), n=len(v))
res["share_persona"] = {"refusal": dist([share(Qp, ref)]), "dmu": dist([share(Qp, dmu_all)]),
                        "shuf_refusal": dist([share(Qp, perm_dir(XR, yR)) for _ in range(200)]),
                        "shuf_dmu": dist([share(Qp, perm_dir(Xh, yh)) for _ in range(200)]),
                        "strat_refusal": dist([share(Qp, strat_dir(XR, yR)) for _ in range(200)]),
                        "strat_dmu": dist([share(Qp, strat_dir(Xh, yh)) for _ in range(200)]),
                        "shuf_alpaca": dist([share(Qp, v) for v in cb["noise_d0"].astype(np.float64)]),
                        "random_direction": dist([share(Qp, rng.standard_normal(5376)) for _ in range(200)])}

Qr = SPACES["random"]
def variants(d):
    pin = Qp @ (Qp.T @ d)
    return {"full": unit(d), "persona_component": unit(pin), "persona_removed": unit(d - pin),
            "random_removed": [unit(d - Q @ (Q.T @ d)) for Q in Qr]}

def scores(V):  # V: (k, 5376) directions -> per-direction (HB per split [50], transfer mean)
    hbs = np.array([fold(rank_auc(yh[te], Xh[te] @ V.T)) for _, te in splits])  # (50, k)
    xf = np.mean([fold(rank_auc(y, X @ V.T)) for X, y in fam.values()], axis=0)    # (k,)
    return hbs, xf

def summarize(hb_split, xf):  # hb_split: (50,) mean over draws per split
    return dict(hb=float(hb_split.mean()), hb_ci=float(1.96 * hb_split.std(ddof=1) / np.sqrt(len(hb_split))), transfer=float(np.mean(xf)))

def fixed_direction(dirs):  # dirs: list of directions (draws); same direction for every split
    out = {}
    vs = [variants(d) for d in dirs]
    for k in ("full", "persona_component", "persona_removed"):
        hbs, xf = scores(np.stack([v[k] for v in vs])); out[k] = summarize(hbs.mean(1), xf)
    hbs, xf = scores(np.stack([w for v in vs for w in v["random_removed"]])); out["random_removed"] = summarize(hbs.mean(1), xf)
    return out

def split_direction(make, n_draw):  # direction re-estimated on the train rows of each split
    acc = {k: ([], []) for k in ("full", "persona_component", "persona_removed", "random_removed")}
    for si, (tr, te) in enumerate(splits):
        hb_k, xf_k = {k: [] for k in acc}, {k: [] for k in acc}
        for _ in range(n_draw):
            v = variants(make(tr))
            for k in acc:
                W = np.stack(v[k]) if k == "random_removed" else v[k][None]
                hb_k[k] += list(fold(rank_auc(yh[te], Xh[te] @ W.T)))
                xf_k[k] += list(np.mean([fold(rank_auc(y, X @ W.T)) for X, y in fam.values()], axis=0))
        for k in acc: acc[k][0].append(np.mean(hb_k[k])); acc[k][1].append(np.mean(xf_k[k]))
    return {k: summarize(np.array(h), np.array(x)) for k, (h, x) in acc.items()}

res["auc"]["refusal"] = fixed_direction([ref])
res["auc"]["dmu"] = split_direction(lambda tr: Xh[tr][yh[tr] == 1].mean(0) - Xh[tr][yh[tr] == 0].mean(0), 1)
res["auc"]["shuf_refusal"] = fixed_direction([perm_dir(XR, yR) for _ in range(20)])
res["auc"]["shuf_dmu"] = split_direction(lambda tr: (lambda p: Xh[tr][p == 1].mean(0) - Xh[tr][p == 0].mean(0))(rng.permutation(yh[tr])), 10)
res["auc"]["strat_refusal"] = fixed_direction([strat_dir(XR, yR) for _ in range(20)])
res["auc"]["strat_dmu"] = split_direction(lambda tr: (lambda p: Xh[tr][p == 1].mean(0) - Xh[tr][p == 0].mean(0))(strat_labels(yh[tr])), 10)
res["auc"]["shuf_alpaca"] = fixed_direction(list(cb["noise_d0"].astype(np.float64)[:20]))
res["auc"]["random_direction"] = fixed_direction([rng.standard_normal(5376) for _ in range(20)])
json.dump(res, open(HERE / "appendix_controls_data.json", "w"), indent=1)

print("share of refusal / dmu in each 240-dim space")
for sn in SPACES: print(f"  {sn:14s} refusal {res['share']['refusal'][sn]['mean']:.3f}  dmu {res['share']['dmu'][sn]['mean']:.3f}")
print("share in the trait span"); [print(f"  {k:16s} {v['mean']:.3f} [{v['p5']:.3f}, {v['p95']:.3f}]") for k, v in res["share_persona"].items()]
print("folded AUC: HarmBench / transfer")
for dn, V in res["auc"].items():
    print(f"  {dn:16s} " + "  ".join(f"{k} {v['hb']:.3f}/{v['transfer']:.3f}" for k, v in V.items()))
