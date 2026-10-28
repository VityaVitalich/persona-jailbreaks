#!/usr/bin/env python3
"""
Part 1: is the PersonaJB detector a role-play detector? Uses only saved outputs:
  roleplay_labels_v2.jsonl (gpt-4.1-mini, classify_roleplay.py rubric v2; v1 kept for agreement) joined to prompts.jsonl, and the
  50-seed PersonaJB per-prompt scores (trait_pc_sweep/scores_llama.npz, config traits_k229 =
  published row, reproduces tab:detection exactly).

Per set (HB test, GCG, PAIR, PAP, GPTFuzz, PEZ, pooled transfer; PAIR also attacks-only):
  share of role-play prompts (strict = persona, broad = persona|fiction), jailbreak rate by stratum,
  detector AUC overall / inside role-play / inside non-role-play,
  AUC of the role-play label alone as a jailbreak predictor,
  AUC of the detector for role-play vs not among NON-jailbroken prompts (does it fire on role-play
  per se when nothing was jailbroken?).
AUCs are signed (no max(a,1-a) flip). Transfer sets: AUC of each seed's refit model, mean over
seeds; 95% CI = behaviour-clustered bootstrap (1000) of the mean-over-seeds AUC. HB test: each
seed's own test split, mean and 1.96*sd/sqrt(n) over seeds.
Writes part1_results.json and part1_tables.md; WJB (not in the Llama detection table) is only
counted, not scored.
"""
import hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent / "trait_pc_sweep/scores_llama.npz"
CFG = "traits_k229"
FAMS = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
rng = np.random.default_rng(0)


def auc(y, s):
    y = np.asarray(y); return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auc_rows(y, S):
    """Signed AUC of every row of S [k, n] against binary y (rank formula, ties averaged)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool); npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0: return np.full(len(S), np.nan)
    R = rankdata(S, axis=1)
    return (R[:, y].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)


def boot_ci(y, S, groups, B=1000):
    """95% CI of the mean-over-seeds AUC; behaviour-clustered bootstrap over prompts."""
    y, S, groups = np.asarray(y), np.asarray(S), np.asarray(groups)
    if not 0 < y.sum() < len(y): return [float("nan")] * 2
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(B):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[ii].sum() < len(ii): vals.append(np.nanmean(auc_rows(y[ii], S[:, ii])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def strata_metrics(y, S, rp, beh):
    """y [n], S [seeds, n] scores, rp [n] bool, beh [n] cluster ids."""
    out = {"n": int(len(y)), "n_rp": int(rp.sum()), "rp_share": float(rp.mean()),
                          "jb_rate_rp": float(y[rp].mean()) if rp.any() else None,
                          "jb_rate_nonrp": float(y[~rp].mean()) if (~rp).any() else None}
    for name, m in [("all", np.ones_like(rp)), ("within_rp", rp), ("within_nonrp", ~rp)]:
        per_seed = [auc(y[m], S[k][m]) for k in range(len(S))] if m.sum() else [float("nan")]
        out[f"auc_{name}"] = float(np.nanmean(per_seed)) if not np.all(np.isnan(per_seed)) else float("nan")
        out[f"auc_{name}_ci"] = boot_ci(y[m], S[:, m], beh[m]) if m.sum() else [float("nan")] * 2
        out[f"n_{name}"] = int(m.sum()); out[f"pos_{name}"] = int(y[m].sum())
    out["auc_rp_label_only"] = auc(y, rp.astype(float))
    neg = y == 0  # detector score: role-play vs not, among non-jailbroken prompts
    out["auc_score_rp_vs_nonrp_among_nonjb"] = float(np.nanmean([auc(rp[neg], S[k][neg]) for k in range(len(S))])) if 0 < rp[neg].sum() < neg.sum() else float("nan")
    return out


def main():
    prompts = [json.loads(l) for l in open(HERE / "prompts.jsonl")]
    def read_labels(fn):
        lab = {}
        for l in open(HERE / fn):
            d = json.loads(l); lab[d["hash"]] = d["label"]
        return lab
    labels, labels_v1 = read_labels("roleplay_labels_v2.jsonl"), read_labels("roleplay_labels.jsonl")
    for r in prompts:
        h = hashlib.sha1(r["prompt"].encode("utf-8")).hexdigest()
        r["rp_label"], r["rp_label_v1"] = labels.get(h, "MISSING"), labels_v1.get(h, "MISSING")
    agree = {s: {"3class": float(np.mean([r["rp_label"] == r["rp_label_v1"] for r in prompts if r["set"] == s])),
                 "broad": float(np.mean([(r["rp_label"] != "none") == (r["rp_label_v1"] != "none") for r in prompts if r["set"] == s]))}
             for s in ["HB"] + FAMS + ["WJB"]}
    print("v1/v2 agreement:", json.dumps(agree))
    lab_counts = {s: dict(Counter(r["rp_label"] for r in prompts if r["set"] == s)) for s in ["HB"] + FAMS + ["WJB"]}
    bad = sum(v for c in lab_counts.values() for k, v in c.items() if k not in ("persona", "fiction", "none"))
    print("label counts:", json.dumps(lab_counts)); print("missing/error labels:", bad)

    d = np.load(NPZ, allow_pickle=True)
    by = {(r["set"], r["pair_id"]): r for r in prompts}
    res = {"label_counts": lab_counts, "n_bad_labels": bad, "labels": "roleplay_labels_v2.jsonl (rubric v2)", "v1_v2_agreement": agree, "definitions": {"strict": "persona", "broad": "persona|fiction"}}
    pooled = {k: {"y": [], "S": [], "rp": [], "beh": []} for k in ("strict", "broad")}

    def rows_for(s):
        pids = d[f"pair_ids__{s}"]; rr = [by[(s, int(p))] for p in pids]
        y = d[f"labels__{s}"].astype(int)
        assert all(int(r["jailbroken"]) == int(t) for r, t in zip(rr, y)), s
        return rr, y

    # HB test: per-seed own test split
    rr, y = rows_for("HB"); S = d[f"scores__{CFG}__HB_refit"]; seeds = d["seeds"]
    res["HB_test"] = {}
    for defn, keep in [("strict", {"persona"}), ("broad", {"persona", "fiction"})]:
        rp = np.array([r["rp_label"] in keep for r in rr]); per = {k: [] for k in ("all", "within_rp", "within_nonrp")}
        neg_auc = []
        for si, seed in enumerate(seeds):
            te = d[f"idx_test__s{seed}"]; yt, st, rt = y[te], S[si][te], rp[te]
            per["all"].append(auc(yt, st)); per["within_rp"].append(auc(yt[rt], st[rt]) if rt.any() else np.nan)
            per["within_nonrp"].append(auc(yt[~rt], st[~rt]) if (~rt).any() else np.nan)
            ng = yt == 0; neg_auc.append(auc(rt[ng], st[ng]) if 0 < rt[ng].sum() < ng.sum() else np.nan)
        o = {"rp_share": float(rp.mean()), "n_templates_rp": len({r["template"] for r, f in zip(rr, rp) if f}),
             "jb_rate_rp": float(y[rp].mean()), "jb_rate_nonrp": float(y[~rp].mean()) if (~rp).any() else None,
             "auc_rp_label_only": auc(y, rp.astype(float)),
             "auc_score_rp_vs_nonrp_among_nonjb": float(np.nanmean(neg_auc))}
        for k, v in per.items():
            v = np.array(v, float); ok = ~np.isnan(v)
            o[f"auc_{k}"] = float(v[ok].mean()); o[f"auc_{k}_ci"] = [float(v[ok].mean() - 1.96 * v[ok].std(ddof=1) / np.sqrt(ok.sum())),
                                                                      float(v[ok].mean() + 1.96 * v[ok].std(ddof=1) / np.sqrt(ok.sum()))]
            o[f"n_seeds_{k}"] = int(ok.sum())
        res["HB_test"][defn] = o

    for s in FAMS + ["PAIR_attacks_only"]:
        fam = "PAIR" if s == "PAIR_attacks_only" else s
        rr, y = rows_for(fam); S = d[f"scores__{CFG}__{fam}"]
        m = np.array([r["attack_type"] != "plain" for r in rr]) if s == "PAIR_attacks_only" else np.ones(len(rr), bool)
        rr = [r for r, k in zip(rr, m) if k]; y, S = y[m], S[:, m]
        beh = np.array([r["behavior_id"] for r in rr]); res[s] = {}
        for defn, keep in [("strict", {"persona"}), ("broad", {"persona", "fiction"})]:
            rp = np.array([r["rp_label"] in keep for r in rr])
            res[s][defn] = strata_metrics(y, S, rp, beh)
            if s in FAMS:
                p = pooled[defn]; p["y"].append(y); p["S"].append(S); p["rp"].append(rp); p["beh"].append(np.array([f"{s}:{b}" for b in beh]))
    res["pooled_transfer"] = {k: strata_metrics(np.concatenate(p["y"]), np.concatenate(p["S"], 1), np.concatenate(p["rp"]), np.concatenate(p["beh"]))
                              for k, p in pooled.items()}
    json.dump(res, open(HERE / "part1_results.json", "w"), indent=1)

    f = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    ci = lambda c: "" if np.isnan(c[0]) else f" [{c[0]:.2f},{c[1]:.2f}]"
    lines = []
    for defn in ("broad", "strict"):
        lines += [f"\n### Role-play = {res['definitions'][defn]} ({defn})\n",
                  "| Set | n | RP share | JB rate RP / non-RP | AUC all | AUC within RP | AUC within non-RP | AUC of RP label alone | Score AUC RP vs non-RP (non-jailbroken only) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in ["HB_test"] + FAMS + ["PAIR_attacks_only", "pooled_transfer"]:
            o = res[s][defn]
            lines.append(f"| {s} | {o.get('n', '')} | {o['rp_share']:.2f} | {f(o['jb_rate_rp'])} / {f(o['jb_rate_nonrp'])} | "
                         f"{f(o['auc_all'])}{ci(o['auc_all_ci'])} | {f(o['auc_within_rp'])}{ci(o['auc_within_rp_ci'])} | "
                         f"{f(o['auc_within_nonrp'])}{ci(o['auc_within_nonrp_ci'])} | {f(o['auc_rp_label_only'])} | {f(o['auc_score_rp_vs_nonrp_among_nonjb'])} |")
    open(HERE / "part1_tables.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
