"""Does PersonaJB catch jailbreaks the raw probe misses, beyond what a random-direction detector would? Saved scores only
(no refits): PersonaJB / raw / Gaussian0-2 test scores of persona_four (single-set training, C tuned on the training set's
val split; Llama + Qwen from scores_cross_family_tuned.npz, Gemma + OLMo from scores_four_models.npz + scores_persona_four.npz).
For every (train a, test family b != a): each score oriented by its folded sign; a detector "flags" a prompt when its score
exceeds the 90th percentile of the non-jailbroken prompts of b (10% FPR; uses test labels, diagnostic only).
Reported per model, pooled over all (a, b):
  catch table on jailbroken prompts (both / PersonaJB only / raw only / neither) and the same with Gaussian in place of PersonaJB;
  rescue = recall among jailbroken prompts the raw probe misses; precision of the extra flags (PersonaJB-only flags that are
  jailbroken / all PersonaJB-only flags);
  stable rescues = jailbroken test prompts flagged by PersonaJB and missed by raw under >= 2/3 of their training sets;
  untuned score average (mean within-b percentile rank of raw and PersonaJB vs raw and Gaussian), avg folded AUC with
  behaviour-cluster bootstrap (B=1000, seed 12345), paired deltas, Bonferroni over 4 models x 2 contrasts;
  trait profile of stable rescues: per-trait projection z-scored on b's non-jailbroken prompts, mean over rescues (top traits).
Saves per-prompt flags/ranks to complement.npz, tables to complement.json."""
import json, sys, numpy as np, collections
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".."); import olmo_probe_lib as L
SETS, FAM, G, B = L.SETS, L.FAM, 3, 1000
FZ = np.load("../scores_four_models.npz"); PZ = np.load("../scores_persona_four.npz")
CZ = np.load("/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b/transfer/diag/scores_cross_family_tuned.npz")
def get(Z, m, det, a, b): return Z[[k for k in Z.files if k.startswith(f"{m}__{det}__train_{a}__test_{b}__C")][0]].astype(np.float64)
def scores(m, a, b):
    if m in ("llama", "qwen"):
        out = {"PersonaJB": get(CZ, m, "PersonaJB", a, b), "raw": get(CZ, m, "fullprobe", a, b)}; Z = CZ
    else:
        out = {"PersonaJB": get(PZ, m, "PersonaJB", a, b), "raw": get(FZ, m, "raw", a, b)}; Z = FZ
    for g in range(G): out[f"G{g}"] = get(Z, m, f"Gaussian{g}", a, b)
    return out
pct = lambda s: (np.argsort(np.argsort(s)) + 0.5) / len(s)
res, SC = {}, {}
for m in ("llama", "gemma", "qwen", "olmo"):
    D = {s: L.LOADERS[m](s) for s in SETS}; Z = CZ if m in ("llama", "qwen") else FZ
    for s in SETS: assert (Z[f"{m}__labels_{s}"] == D[s][1]).all(), (m, s)
    T, tnames = L.trait_matrix(m)
    cnt = collections.Counter(); cntG = collections.Counter(); perfam = collections.defaultdict(collections.Counter)
    neg_extra = collections.Counter(); uniq = {b: np.zeros((len(D[b][1]), 2 + G)) for b in FAM}; ntr = {b: 0 for b in FAM}
    ens = {k: {} for k in ("raw", "raw+PersonaJB", "raw+Gaussian")}
    for b in FAM:
        y = D[b][1]
        for a in [x for x in SETS if x != b]:
            S = scores(m, a, b); ntr[b] += 1
            for k in S: S[k] = S[k] * (1 if roc_auc_score(y, S[k]) >= .5 else -1)
            F = {k: S[k] > np.quantile(S[k][y == 0], .9) for k in S}
            p, r = F["PersonaJB"], F["raw"]
            for name, c, x in (("PersonaJB", cnt, p), *[(f"G{g}", cntG, F[f"G{g}"]) for g in range(G)]):
                w = 1 if name == "PersonaJB" else 1 / G
                c["both"] += w * (x & r & (y == 1)).sum(); c["only_alt"] += w * (x & ~r & (y == 1)).sum(); c["only_raw"] += w * (~x & r & (y == 1)).sum()
                c["neither"] += w * (~x & ~r & (y == 1)).sum(); c["alt_only_neg"] += w * (x & ~r & (y == 0)).sum()
                if name == "PersonaJB": perfam[b]["P_only"] += (x & ~r & (y == 1)).sum(); perfam[b]["raw_miss"] += (~r & (y == 1)).sum()
                else: perfam[b]["G_only"] += w * (x & ~r & (y == 1)).sum()
            uniq[b][:, 0] += p & ~r; uniq[b][:, 1] += r
            for g in range(G): uniq[b][:, 2 + g] += F[f"G{g}"] & ~r
            rr, rp = pct(S["raw"]), pct(S["PersonaJB"]); rg = np.mean([pct(S[f"G{g}"]) for g in range(G)], 0)
            ens["raw"][(a, b)], ens["raw+PersonaJB"][(a, b)], ens["raw+Gaussian"][(a, b)] = rr, (rr + rp) / 2, (rr + rg) / 2
            for k, v in (("rawPJB", (rr + rp) / 2), ("rawG", (rr + rg) / 2)): SC[f"{m}__ens_{k}__train_{a}__test_{b}"] = v.astype(np.float32)
    # catch tables
    def tab(c):
        miss = c["only_alt"] + c["neither"]; return {"both": c["both"], "alt_only": c["only_alt"], "raw_only": c["only_raw"], "neither": c["neither"],
            "rescue_recall": c["only_alt"] / miss, "extra_flag_precision": c["only_alt"] / (c["only_alt"] + c["alt_only_neg"])}
    r_ = {"PersonaJB_vs_raw": tab(cnt), "Gaussian_vs_raw": tab(cntG),
          "per_family_rescue": {b: {"PersonaJB": perfam[b]["P_only"] / perfam[b]["raw_miss"], "Gaussian": perfam[b]["G_only"] / perfam[b]["raw_miss"], "raw_missed": int(perfam[b]["raw_miss"])} for b in FAM}}
    # stable rescues: flagged by PersonaJB and missed by raw under >= 2/3 of the training sets
    stab, stabG, prof, cats = 0, 0.0, [], collections.Counter()
    for b in FAM:
        y = D[b][1]; k = ntr[b]; sp = (uniq[b][:, 0] >= 2 * k / 3) & (y == 1); stab += sp.sum()
        stabG += np.mean([((uniq[b][:, 2 + g] >= 2 * k / 3) & (y == 1)).sum() for g in range(G)])
        SC[f"{m}__stable_rescue_{b}"] = sp; SC[f"{m}__pjb_only_count_{b}"] = uniq[b][:, 0].astype(np.int8); SC[f"{m}__pair_ids_{b}"] = np.array([str(r["pair_id"]) for r in D[b][2]])
        if sp.any():
            Pj = D[b][0] @ T.T; mu, sd = Pj[y == 0].mean(0), Pj[y == 0].std(0) + 1e-9
            prof.append(((Pj[sp] - mu) / sd)); cats.update(D[b][2][i].get("semantic_category") for i in np.where(sp)[0])
            # reference: jailbroken prompts the raw probe catches in most training sets
            rc = (uniq[b][:, 1] >= 2 * k / 3) & (y == 1)
            if rc.any(): r_.setdefault("_ref", []).append((Pj[rc] - mu) / sd)
    r_["stable_rescues"] = {"PersonaJB": int(stab), "Gaussian_mean": float(stabG), "n_jailbroken_test": int(sum(D[b][1].sum() for b in FAM)), "semantic_category": dict(cats)}
    if prof:
        Pz = np.vstack(prof); mz = Pz.mean(0); o = np.argsort(-np.abs(mz))[:8]
        ref = np.vstack(r_.pop("_ref")).mean(0) if "_ref" in r_ else np.zeros_like(mz)
        r_["rescue_trait_profile_top"] = [{"trait": tnames[i], "z_rescued": float(mz[i]), "z_raw_caught": float(ref[i])} for i in o]
        r_["rescue_profile_vs_raw_caught_cos"] = float(mz @ ref / np.linalg.norm(mz) / np.linalg.norm(ref))
    r_.pop("_ref", None)
    # untuned score average, avg folded AUC over (a, b) like the persona_four avg; paired bootstrap
    behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {x: i for i, x in enumerate(behs)}
    W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)
    E = {}
    for k, V in ens.items():
        per_a = []
        for a in SETS:
            pb = [L.boot_auc(D[b][1], V[(a, b)], W[:, [bi[r["behavior_id"]] for r in D[b][2]]]) for b in FAM if b != a]
            per_a.append((np.mean([x[0] for x in pb]), np.nanmean(np.stack([x[1] for x in pb]), 0)))
        E[k] = (np.mean([x[0] for x in per_a]), np.mean([x[1] for x in per_a], 0)); SC[f"{m}__boot_{k}"] = E[k][1].astype(np.float32)
    zq = 1 - 0.05 / 8 / 2; dci = lambda d: [float(np.nanpercentile(d, 100 * (1 - zq))), float(np.nanpercentile(d, 100 * zq))]
    r_["score_average_auc"] = {k: float(v[0]) for k, v in E.items()}
    r_["score_average_delta"] = {"raw+PersonaJB - raw": [float(E["raw+PersonaJB"][0] - E["raw"][0]), dci(E["raw+PersonaJB"][1] - E["raw"][1])],
                                 "raw+PersonaJB - raw+Gaussian": [float(E["raw+PersonaJB"][0] - E["raw+Gaussian"][0]), dci(E["raw+PersonaJB"][1] - E["raw+Gaussian"][1])]}
    res[m] = r_
    print(m, json.dumps({k: v for k, v in r_.items() if k not in ("rescue_trait_profile_top",)}, default=float)[:1500], flush=True)
    print("  top traits of stable rescues:", [(t["trait"], round(t["z_rescued"], 2), round(t["z_raw_caught"], 2)) for t in r_.get("rescue_trait_profile_top", [])], flush=True)
json.dump(res, open("complement.json", "w"), indent=1, default=float); np.savez("complement.npz", **SC)
