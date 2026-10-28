"""Raw-activation linear probe on OLMo-3-7B-Instruct, same protocol as qwen3_32b/transfer/diag/cross_family_tuned.py
(StandardScaler + LR; C from GRID chosen on a behaviour-grouped validation split of the TRAINING set, HB also
template-disjoint; refit on the whole training set; folded AUC on every other set; cluster bootstrap over HarmBench
behaviours, B=1000, shared across test sets, orientation fixed at the point estimate). Transfer = mean AUC over
GCG/PAIR/PAP/GPTFuzz/PEZ minus the training set.
Main readout: layer 16 of 32 (Llama's L16 of 32), last prompt token. Also: the same transfer at every saved layer (no
bootstrap), and within-set AUC (5-fold GroupKFold by behaviour, C tuned on a validation split inside each training fold).
Saves every per-prompt score (val scores for every C, test scores, within-set out-of-fold scores, layer sweep test
scores) to scores_olmo_probe.npz; table to olmo_probe.json."""
from olmo_probe_lib import *  # noqa: F401,F403  (SETS, FAM, TAG, load, cell, within, ...)
D = {s: load(s) for s in SETS}; LAYERS = D["hb"][3]; Lmain = LAYERS.index(MAIN)
print({s: f"{D[s][1].sum()}/{len(D[s][1])}" for s in SETS}, flush=True)
jobs = [delayed(cell)(li, a, D[a][0][:, li].astype(np.float64), D[a][1], D[a][2], {b: D[b][0][:, li].astype(np.float64) for b in SETS if b != a}) for li in range(len(LAYERS)) for a in SETS]
OUT = dict(Parallel(n_jobs=8, backend="loky")(jobs)); print("transfer fits done", flush=True)
WI = dict(Parallel(n_jobs=7, backend="loky")(delayed(within)(a, D[a][0][:, Lmain].astype(np.float64), D[a][1], D[a][2]) for a in SETS)); print("within fits done", flush=True)
# cluster bootstrap over behaviours (main layer), shared across test sets
rng_b = np.random.default_rng(12345)
behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {b: i for i, b in enumerate(behs)}
W = rng_b.multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B); bidx = {s: np.array([bi[r["behavior_id"]] for r in D[s][2]]) for s in SETS}
def auc_boot(y, s, b_of):
    sign = 1 if roc_auc_score(y, s) >= 0.5 else -1; pt = roc_auc_score(y, sign * s)
    bs = np.array([roc_auc_score(y, sign * s, sample_weight=W[k][b_of]) if (W[k][b_of] * y).sum() > 0 and (W[k][b_of] * (1 - y)).sum() > 0 else np.nan for k in range(B)])
    return pt, bs
res, SC = {"main_layer": MAIN, "layers": LAYERS, "grid": GRID, "B": B, "transfer": {}, "per_family": {}, "within": {}, "layer_sweep": {}}, {}
avg_bs = []
for a in SETS:
    fams = [f for f in FAM if f != a]; pts, bss = [], []
    for f in fams: pt, bs = auc_boot(D[f][1], OUT[(Lmain, a)]["test"][f], bidx[f]); pts.append(pt); bss.append(bs)
    bm = np.nanmean(np.stack(bss), 0); avg_bs.append(bm); SC[f"boot__train_{a}"] = bm.astype(np.float32)
    res["transfer"][a] = {"auc": float(np.mean(pts)), "ci95": [float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))], "C": OUT[(Lmain, a)]["C"]}
    res["per_family"][a] = dict(zip(fams, map(float, pts)))
    pt, bs = auc_boot(D[a][1], WI[a]["oof"], bidx[a]); res["within"][a] = {"auc": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "C_folds": WI[a]["C"]}
avg = np.mean(np.stack(avg_bs), 0); res["transfer"]["avg"] = {"auc": float(np.mean([res["transfer"][a]["auc"] for a in SETS])), "ci95": [float(np.nanpercentile(avg, 2.5)), float(np.nanpercentile(avg, 97.5))]}
for li, l in enumerate(LAYERS):
    res["layer_sweep"][str(l)] = {a: float(np.mean([fold(D[f][1], OUT[(li, a)]["test"][f]) for f in FAM if f != a])) for a in SETS}
    res["layer_sweep"][str(l)]["avg"] = float(np.mean(list(res["layer_sweep"][str(l)].values())))
for (li, a), v in OUT.items():
    p = f"L{LAYERS[li]}__train_{a}"
    for b, s in v["test"].items(): SC[f"{p}__test_{b}__C{v['C']:g}"] = s
    if li == Lmain:
        SC[f"{p}__val_idx"] = v["val_idx"]
        for C, s in v["val_scores"].items(): SC[f"{p}__val_C{C:g}"] = s
for a in SETS:
    SC[f"within_L{MAIN}__{a}__oof"] = WI[a]["oof"]; SC[f"labels_{a}"] = D[a][1]
    SC[f"pair_ids_{a}"] = np.array([r["pair_id"] for r in D[a][2]]); SC[f"behavior_ids_{a}"] = np.array([r["behavior_id"] for r in D[a][2]])
json.dump(res, open(f"olmo_probe{TAG}.json", "w"), indent=1); np.savez(f"scores_olmo_probe{TAG}.npz", **SC)
print("transfer L16 " + " | ".join(f"{a}: {res['transfer'][a]['auc']:.3f} [{res['transfer'][a]['ci95'][0]:.3f},{res['transfer'][a]['ci95'][1]:.3f}]" for a in SETS + ["avg"]))
print("within   L16 " + " | ".join(f"{a}: {res['within'][a]['auc']:.3f}" for a in SETS))
for l in LAYERS: print(f"sweep L{l:2d} avg {res['layer_sweep'][str(l)]['avg']:.3f}")
