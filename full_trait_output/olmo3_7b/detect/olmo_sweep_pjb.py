"""OLMo-3-7B layer x token-position sweep: PersonaJB (240 unfiltered trait vectors at the SAME layer, pre-generation last
token) vs raw-activation LR, single-set training, at every saved readout: layers 4,8,...,28,31 x the 5 chat-template tail
positions (<|im_end|> \\n <|im_start|> assistant \\n; -1 = the position the trait vectors were extracted at).
Protocol per cell and training set as olmo_probe_lib.cell (C tuned on a behaviour-grouped validation split of the training set).
Cell selection without test data: per detector and training set, the readout with the highest best-C validation AUC
("val-selected"); bootstrap (B=1000, behaviour clusters) for the val-selected readouts and for the fixed L16 / last-token one,
paired delta PersonaJB - raw, Bonferroni over 2 x 8 columns. Saves every val/test score and bootstrap draw to
scores_olmo_sweep_pjb.npz, table olmo_sweep_pjb.json.   usage: OLMO_TR=... python olmo_sweep_pjb.py TRAIT_DIR OUTDIR NJOBS"""
import glob, json, os, sys, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
import olmo_probe_lib as L
TD, OUT, NJ = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(OUT, exist_ok=True)
SETS, FAM, B = L.SETS, L.FAM, 1000
P = sorted(glob.glob(f"{TD}/*.pt")); TA = np.stack([torch.load(p, map_location="cpu", weights_only=False)["vector"].float().numpy() for p in P])  # [240, 32, 4096]
D = {}
for s in SETS:
    A = torch.load(f"{L.TR}/acts_{s}.pt"); C = {r["pair_id"]: r for r in map(json.loads, open(f"{L.TR}/classified_{s}.jsonl"))}
    k = [i for i, p in enumerate(A["pair_ids"]) if C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")]
    R = [C[A["pair_ids"][i]] for i in k]; D[s] = (A["X"][k].numpy(), np.array([int(r["jailbroken"]) for r in R]), R)   # fp16 [n, 8, 5, 4096]
    LAYERS, TAIL = A["layers"], A["tail_tokens"]
NPOS = len(TAIL); print("layers", LAYERS, "tail", TAIL, {s: f"{D[s][1].sum()}/{len(D[s][1])}" for s in SETS}, flush=True)
def job(li, p, det, a):
    l = LAYERS[li]; T = TA[:, l]; T = T / np.linalg.norm(T, axis=1, keepdims=True)
    f = (lambda X: X) if det == "raw" else (lambda X: X @ T.T)
    X = {s: f(D[s][0][:, li, p].astype(np.float64)) for s in SETS}
    key, v = L.cell((l, p - NPOS, det), a, X[a], D[a][1], D[a][2], {b: X[b] for b in SETS if b != a})
    return key, v
cells = [(li, p) for li in range(len(LAYERS)) for p in range(NPOS)]
R_ = dict(Parallel(n_jobs=NJ, backend="loky", verbose=2)(delayed(job)(li, p, det, a) for li, p in cells for det in ("PersonaJB", "raw") for a in SETS))
fold = L.fold; res = {"layers": LAYERS, "tail_tokens": TAIL, "grid": {}, "val_selected": {}, "fixed_L16_last": {}}; SC = {}
for li, p in cells:
    l, pos = LAYERS[li], p - NPOS
    for det in ("PersonaJB", "raw"):
        row = {}
        for a in SETS:
            v = R_[((l, pos, det), a)]; row[a] = {"transfer": float(np.mean([fold(D[f][1], v["test"][f]) for f in FAM if f != a])), "val_auc": float(max(v["val_auc"].values())), "C": v["C"]}
            q = f"{det}__L{l}__pos{pos}__train_{a}"; SC[f"{q}__val_idx"] = v["val_idx"]
            for C, s in v["val_scores"].items(): SC[f"{q}__val_C{C:g}"] = s
            for b, s in v["test"].items(): SC[f"{q}__test_{b}__C{v['C']:g}"] = s
        row["avg"] = float(np.mean([row[a]["transfer"] for a in SETS])); res["grid"][f"{det}__L{l}__pos{pos}"] = row
behs = sorted({r["behavior_id"] for s in SETS for r in D[s][2]}); bi = {b: i for i, b in enumerate(behs)}
W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B); bidx = {s: np.array([bi[r["behavior_id"]] for r in D[s][2]]) for s in SETS}
zq = 1 - 0.05 / 16 / 2
for mode in ("val_selected", "fixed_L16_last"):
    Tb = {}
    for det in ("PersonaJB", "raw"):
        for a in SETS:
            if mode == "fixed_L16_last": l, pos = 16, -1
            else: l, pos = max(((LAYERS[li], p - NPOS) for li, p in cells), key=lambda c: res["grid"][f"{det}__L{c[0]}__pos{c[1]}"][a]["val_auc"])
            v = R_[((l, pos, det), a)]; pts, bss = zip(*[L.boot_auc(D[f][1], v["test"][f], W[:, bidx[f]]) for f in FAM if f != a])
            Tb[(det, a)] = (np.mean(pts), np.nanmean(np.stack(bss), 0), f"L{l}/pos{pos}")
        Tb[(det, "avg")] = (np.mean([Tb[(det, a)][0] for a in SETS]), np.mean([Tb[(det, a)][1] for a in SETS], 0), "")
    for a in SETS + ["avg"]:
        d = Tb[("PersonaJB", a)][1] - Tb[("raw", a)][1]
        res[mode][a] = {**{det: float(Tb[(det, a)][0]) for det in ("PersonaJB", "raw")}, **{f"{det}_readout": Tb[(det, a)][2] for det in ("PersonaJB", "raw")},
                        "delta": float(Tb[("PersonaJB", a)][0] - Tb[("raw", a)][0]), "delta_ci_bonf": [float(np.nanpercentile(d, 100 * (1 - zq))), float(np.nanpercentile(d, 100 * zq))]}
        for det in ("PersonaJB", "raw"): SC[f"boot__{mode}__{det}__{a}"] = Tb[(det, a)][1].astype(np.float32)
for s in SETS: SC[f"labels_{s}"] = D[s][1]; SC[f"behavior_ids_{s}"] = np.array([r["behavior_id"] for r in D[s][2]]); SC[f"pair_ids_{s}"] = np.array([r["pair_id"] for r in D[s][2]])
json.dump(res, open(f"{OUT}/olmo_sweep_pjb.json", "w"), indent=1); np.savez(f"{OUT}/scores_olmo_sweep_pjb.npz", **SC)
for det in ("PersonaJB", "raw"):
    print(f"== {det}: avg transfer, layer x position")
    for l in LAYERS: print(f"  L{l:2d} " + " ".join(f"{res['grid'][f'{det}__L{l}__pos{p - NPOS}']['avg']:.3f}" for p in range(NPOS)))
for mode in ("val_selected", "fixed_L16_last"): print(mode, " ".join(f"{a}: PJB {res[mode][a]['PersonaJB']:.3f} raw {res[mode][a]['raw']:.3f} d {res[mode][a]['delta']:+.3f}" for a in SETS + ["avg"]))
