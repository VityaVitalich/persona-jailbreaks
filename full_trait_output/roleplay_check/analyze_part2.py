#!/usr/bin/env python3
"""
Part 2: synthetic verification. Do HARMFUL role-plays score higher than BENIGN role-plays under the
PersonaJB detector, or does any role-play fire it?

Detector: the published PersonaJB row, refit exactly as trait_pc_sweep.py does (50 seeds, HB
human-jailbreak train pool, StandardScaler + L2 logreg C=10 on trait projections T h). The refit
model of each seed scores the synthetic prompts (synthetic_acts.pt, computed on RCP by
embed_cluster.py); scores are P(jailbreak).

Reported (folded AUC = max(a,1-a) in the tables, signed in the JSON; mean over seeds; 95% CI = bootstrap over JBB behaviours, each resample keeps
a behaviour's harmful and benign versions together):
  harmful vs benign  within plain / each frame type / all role-play / each frame
  role-play shift    benign role-play vs benign plain (does the wrapper alone raise the score?)
  cross              harmful plain vs benign role-play (a pure role-play detector would put benign
                     role-play above the plain harmful request -> AUC < .5)
  paired win rate    share of (behaviour, frame) where harmful version > its matched benign version
Robustness: the same without JBB behaviours sourced from HarmBench (behaviour overlap with training).
Writes part2_results.json, part2_tables.md, part2_scores.npz (per-prompt scores, every seed).
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trait_pc_sweep"))
import trait_pc_sweep as tps  # noqa: E402  (reuse the published-row protocol verbatim)
from analyze_part1 import auc_rows  # noqa: E402

rng = np.random.default_rng(0)
FRAME_TYPES = ["neutral_persona", "neutral_fiction", "jb_persona"]


def fit_scores(Xsyn):
    cfg = tps.CFG["llama"]; L = cfg["layer"]
    T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    hrows = [r for r in tps.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    Xh, yh, hvalid = tps.build_matrix(hrows, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), L)
    W = T.T.astype(np.float32); Zh, Zs = Xh @ W, Xsyn @ W
    S, seeds, hb_auc = [], [], []
    for seed in range(50):
        tr, te = tps.pool_split(hvalid, seed)
        if not tr or not te or len(np.unique(yh[tr])) < 2: continue
        m = tps.model(seed).fit(Zh[tr], yh[tr])
        S.append(m.predict_proba(Zs)[:, 1]); seeds.append(seed)
        hb_auc.append(tps.auc2(yh[te], m.predict_proba(Zh[te])[:, 1])[1])
    print(f"refit {len(seeds)} seeds; HB test safe-AUC {np.nanmean(hb_auc):.4f} (published .8538)", flush=True)
    return np.array(S), seeds, float(np.nanmean(hb_auc))


def cell(S, rows, sel_pos, sel_neg, B=1000):
    """AUC (mean over seeds) of pos vs neg rows + behaviour-clustered bootstrap CI."""
    pos = np.array([sel_pos(r) for r in rows]); neg = np.array([sel_neg(r) for r in rows])
    m = pos | neg; y = pos[m]; Sm = S[:, m]; beh = np.array([r["jbb_index"] for r in rows])[m]
    point = float(np.nanmean(auc_rows(y, Sm)))
    ub = np.unique(beh); idx = {b: np.where(beh == b)[0] for b in ub}; vals = []
    for _ in range(B):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        vals.append(np.nanmean(auc_rows(y[ii], Sm[:, ii])))
    return {"auc": point, "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def paired_win(S, rows, frame_sel):
    key = {(r["harm"], r["jbb_index"], r["frame"]): i for i, r in enumerate(rows)}
    pairs = [(key[("harmful", r["jbb_index"], r["frame"])], key[("benign", r["jbb_index"], r["frame"])])
             for r in rows if r["harm"] == "harmful" and frame_sel(r)]
    h, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    return float((S[:, h] > S[:, b]).mean()), len(pairs)


def analyse(S, rows):
    H, Bn = (lambda r: r["harm"] == "harmful"), (lambda r: r["harm"] == "benign")
    is_rp = lambda r: r["frame_type"] != "plain"
    res = {"harm_vs_benign": {}, "per_frame": {}, "paired_win": {}}
    for name, f in [("plain", lambda r: r["frame_type"] == "plain"), ("all_roleplay", is_rp)] + \
                   [(ft, (lambda ft: lambda r: r["frame_type"] == ft)(ft)) for ft in FRAME_TYPES]:
        res["harm_vs_benign"][name] = cell(S, rows, lambda r, f=f: H(r) and f(r), lambda r, f=f: Bn(r) and f(r))
        res["paired_win"][name] = paired_win(S, rows, f)
    for fr in sorted({r["frame"] for r in rows if is_rp(r)}):
        res["per_frame"][fr] = cell(S, rows, lambda r: H(r) and r["frame"] == fr, lambda r: Bn(r) and r["frame"] == fr, B=200)
        res["per_frame"][fr]["type"] = next(r["frame_type"] for r in rows if r["frame"] == fr)
    res["rp_shift"] = {ft: cell(S, rows, lambda r, ft=ft: Bn(r) and r["frame_type"] == ft, lambda r: Bn(r) and r["frame_type"] == "plain")
                       for ft in FRAME_TYPES + ["all_roleplay"] if ft != "all_roleplay"}
    res["rp_shift"]["all_roleplay"] = cell(S, rows, lambda r: Bn(r) and is_rp(r), lambda r: Bn(r) and not is_rp(r))
    res["cross_harmplain_vs_benignrp"] = {ft: cell(S, rows, lambda r: H(r) and not is_rp(r), lambda r, ft=ft: Bn(r) and r["frame_type"] == ft)
                                          for ft in FRAME_TYPES}
    # two-axis test on the same prompts: a role-play detector separates the role-play axis, not the harm axis
    res["axis_harm_vs_benign"] = cell(S, rows, H, Bn)
    res["axis_roleplay_vs_plain"] = cell(S, rows, is_rp, lambda r: not is_rp(r))
    for h, f in [("harmful", H), ("benign", Bn)]:
        res[f"axis_roleplay_vs_plain__{h}_only"] = cell(S, rows, lambda r, f=f: f(r) and is_rp(r), lambda r, f=f: f(r) and not is_rp(r))
    sm = S.mean(0)
    res["mean_score"] = {f"{h}/{ft}": float(np.mean([sm[i] for i, r in enumerate(rows) if r["harm"] == h and r["frame_type"] == ft]))
                         for h in ("harmful", "benign") for ft in ["plain"] + FRAME_TYPES}
    return res


def main():
    rows = [json.loads(l) for l in open(HERE / "synthetic_prompts.jsonl")]
    acts = torch.load(HERE / "synthetic_acts.pt", map_location="cpu")
    X = np.stack([acts[r["id"]].float().numpy() for r in rows]).astype(np.float32)
    S, seeds, hb = fit_scores(X)
    np.savez_compressed(HERE / "part2_scores.npz", ids=np.array([r["id"] for r in rows]), seeds=np.array(seeds), scores=S.astype(np.float32))
    res = {"n_seeds": len(seeds), "hb_test_safe_auc_reproduced": hb, "all": analyse(S, rows)}
    hb_sourced = {r["jbb_index"] for r in rows if r["harm"] == "harmful" and r["source"] == "TDC/HarmBench"}  # drop both members of the pair
    keep = np.array([r["jbb_index"] not in hb_sourced for r in rows])
    res["excl_harmbench_sourced"] = analyse(S[:, keep], [r for r, k in zip(rows, keep) if k])
    res["excl_harmbench_sourced"]["n_behaviours"] = len({r["jbb_index"] for r, k in zip(rows, keep) if k})
    json.dump(res, open(HERE / "part2_results.json", "w"), indent=1)

    def c(o):  # folded AUC = max(a, 1-a): only separation matters, direction is arbitrary (P(jb) tracks compliance)
        a, lo, hi = o["auc"], *o["ci"]
        if a < .5: a, lo, hi = 1 - a, 1 - hi, 1 - lo
        return f"{a:.2f} [{lo:.2f}, {hi:.2f}]"
    L = []
    for tag in ("all", "excl_harmbench_sourced"):
        r = res[tag]
        L += [f"\n### {tag}" + (f" ({r['n_behaviours']} behaviours)" if tag != "all" else " (100 behaviours)"),
              "\n| Condition | Separation harmful vs benign (folded AUC) | Paired win rate | Mean P(jb) harmful | Mean P(jb) benign |", "|---|---|---|---|---|"]
        for k in ["plain", "neutral_persona", "neutral_fiction", "jb_persona", "all_roleplay"]:
            ms = r["mean_score"]
            mh = ms.get(f"harmful/{k}", float("nan")); mb = ms.get(f"benign/{k}", float("nan"))
            L.append(f"| {k} | {c(r['harm_vs_benign'][k])} | {r['paired_win'][k][0]:.2f} (n={r['paired_win'][k][1]}) | "
                     + (f"{mh:.3f} | {mb:.3f} |" if k != "all_roleplay" else "– | – |"))
        L += ["\n| Axis (all prompts of this subset) | separation (folded AUC) |", "|---|---|",
              f"| harmful vs benign | {c(r['axis_harm_vs_benign'])} |",
              f"| role-play vs plain | {c(r['axis_roleplay_vs_plain'])} |",
              f"| role-play vs plain, harmful only | {c(r['axis_roleplay_vs_plain__harmful_only'])} |",
              f"| role-play vs plain, benign only | {c(r['axis_roleplay_vs_plain__benign_only'])} |"]
        L += ["\n| Benign role-play vs benign plain (wrapper effect) | AUC |", "|---|---|"]
        L += [f"| {k} | {c(v)} |" for k, v in r["rp_shift"].items()]
        L += ["\n| Harmful plain vs benign role-play | AUC |", "|---|---|"]
        L += [f"| {k} | {c(v)} |" for k, v in r["cross_harmplain_vs_benignrp"].items()]
        if tag == "all":
            L += ["\n| Frame | type | Separation harmful vs benign (folded AUC) |", "|---|---|---|"]
            L += [f"| {k} | {v['type']} | {c(v)} |" for k, v in sorted(r["per_frame"].items(), key=lambda kv: (kv[1]["type"], kv[0]))]
    open(HERE / "part2_tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
