#!/usr/bin/env python3
"""
Test 2: few named directions. Each basis is a dictionary of 229 unit directions (persona traits, or a
control from control_bases_llama.npz). Per pool split (same 50 HarmBench splits as PersonaJB), pick the
k directions whose single-direction AUC on the TRAIN rows is farthest from 0.5, then fit the PersonaJB
detector (StandardScaler + LR C=10) on those k projections and score HB test and the 5 transfer families.
A 229-dim span covers the dominant activation directions almost regardless of its content; k <= 10
named directions only work if they are the right ones.

k=1 is the single best train direction (its orientation is fixed on train, so test AUC is signed).
Reported: HB test AUC (signed: orientation learned on train) and avg transfer, signed and folded
(max(auc,1-auc), the paper's metric).
Writes few_directions.json, few_directions.md, few_directions_scores.npz (every prompt x basis x k x seed,
float16, plus selected direction indices).
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402

KS = [1, 3, 5, 10, 20, 229]
FAMS = S.FAMS


def rank_auc(y, Z):  # AUC of every column of Z, vectorised (ties averaged via rankdata)
    from scipy.stats import rankdata
    R = rankdata(Z, axis=0); n1 = y.sum(); n0 = len(y) - n1
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    cfg = S.CFG["llama"]
    T = np.load(cfg["T"]).astype(np.float32); T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = {"persona": T, **{k: v.astype(np.float32) for k, v in np.load(HERE / "control_bases_llama.npz").items()}}
    hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
    Xh, yh, hv = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
    fam = {}
    for f, (rp, ap) in cfg["fams"].items():
        X, y, _ = S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)
        fam[f] = (X, y)
    splits = [tuple(map(np.array, S.pool_split(hv, s))) for s in range(50)]

    res, npz = {}, {"labels__HB": yh, **{f"labels__{f}": y for f, (_, y) in fam.items()}}
    for n, V in B.items():
        Zh = Xh @ V.T; Zf = {f: X @ V.T for f, (X, _) in fam.items()}
        r = {k: {"hb": [], "xf_signed": [], "xf_folded": []} for k in KS}
        sc = {k: np.zeros((50, len(yh) + sum(len(y) for _, y in fam.values())), np.float16) for k in KS}
        sel = {k: np.zeros((50, k), np.int16) for k in KS}
        for si, (tr, te) in enumerate(splits):
            a = rank_auc(yh[tr], Zh[tr]); order = np.argsort(-np.abs(a - 0.5))
            for k in KS:
                idx = order[:k]; sel[k][si] = idx
                if k == 1:
                    sgn = 1.0 if a[idx[0]] >= 0.5 else -1.0
                    s_hb = sgn * Zh[:, idx[0]]; s_f = {f: sgn * Z[:, idx[0]] for f, Z in Zf.items()}
                else:
                    m = S.model(si).fit(Zh[tr][:, idx], yh[tr])
                    s_hb = m.decision_function(Zh[:, idx]); s_f = {f: m.decision_function(Z[:, idx]) for f, Z in Zf.items()}
                r[k]["hb"].append(roc_auc_score(yh[te], s_hb[te]))
                fa = np.array([roc_auc_score(fam[f][1], s_f[f]) for f in FAMS])
                r[k]["xf_signed"].append(float(fa.mean())); r[k]["xf_folded"].append(float(np.maximum(fa, 1 - fa).mean()))
                sc[k][si] = np.concatenate([s_hb] + [s_f[f] for f in FAMS]).astype(np.float16)
        res[n] = r
        for k in KS:
            npz[f"scores__{n}__k{k}"] = sc[k]; npz[f"selected__{n}__k{k}"] = sel[k]
        print(n.ljust(11), " ".join(f"k{k}:{np.mean(r[k]['hb']):.3f}/{np.mean(r[k]['xf_signed']):.3f}" for k in KS), flush=True)
    npz["splits_test"] = np.array([te for _, te in splits], dtype=object)
    np.savez_compressed(HERE / "few_directions_scores.npz", **npz)
    json.dump(res, open(HERE / "few_directions.json", "w"))

    groups = [("Persona traits", ["persona"]), ("229 Alpaca activations", [f"acts_d{i}" for i in range(5)]),
              ("229 shuffled-label contrasts", [f"noise_d{i}" for i in range(5)]), ("Alpaca PCA-229", ["alpaca_pca"]),
              ("HarmBench PCA-229 (no labels)", ["hb_pca"]), ("Random 229", [f"gauss_d{i}" for i in range(5)])]
    L = []
    for metric, title in [("hb", "HarmBench test AUC (signed, orientation from train)"),
                          ("xf_signed", "Avg. transfer AUC, signed"), ("xf_folded", "Avg. transfer AUC, folded (paper metric)")]:
        L += [f"### {title}", "", "| basis | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---:|" * len(KS)]
        for g, ns in groups:
            cells = []
            for k in KS:
                v = np.array([np.mean(res[n][k][metric]) for n in ns])
                cells.append(f"{v.mean():.3f}" + (f" [{v.min():.3f}, {v.max():.3f}]" if len(v) > 1 else ""))
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")
    se = {k: 1.96 * np.std(res["persona"][k]["hb"], ddof=1) / np.sqrt(50) for k in KS}
    L.append("Persona HB 95% CI half-width over splits: " + ", ".join(f"k={k} ±{v:.3f}" for k, v in se.items()))
    L.append("Controls with 5 draws: mean over draws [min, max]. Directions picked by train-split single-direction AUC.")
    open(HERE / "few_directions.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
