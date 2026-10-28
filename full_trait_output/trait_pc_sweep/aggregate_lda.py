#!/usr/bin/env python3
"""Persona-LDA axes vs trait PCA (Llama): avg transfer with C chosen per seed on HB inner val, plus the label-free
persona-identification accuracy of the same axes, and the share of the PersonaJB normal w / jailbreak mean shift
that the span of the top-k axes captures."""
import json, sys, numpy as np, torch
sys.path.insert(0, "."); import trait_pc_sweep as S
FAMS = S.FAMS; CS = ["0.01", "0.1", "1", "10"]
def valC(get):
    val = np.array([get(c)["val"]["auc"] for c in CS]); pick = np.nanargmax(val, 0)
    per = {f: np.array([get(CS[pick[s]])[f]["auc"][s] for s in range(val.shape[1])]) for f in ["human_test"] + FAMS}
    xf = np.mean([per[f] for f in FAMS], 0); return float(np.nanmean(per["human_test"])), float(xf.mean()), float(1.96*xf.std(ddof=1)/np.sqrt(len(xf)))
L = {c: json.load(open(f"summary_llama_lda_C{c}.json"))["configs"] for c in CS}
P = {c: json.load(open(f"summary_llama{'' if c == '10' else '_C' + c}.json"))["configs"] for c in CS}
chk = json.load(open("lda_axes_llama.json"))["persona_id_check"]
B = np.load("lda_bases_llama.npz")
# contrast shares
cfg = S.CFG["llama"]; T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
rows = [x for x in S.load_jsonl(cfg["human"][0]) if x.get("attack_type") == "human_jailbreak"]
X, y, valid = S.build_matrix(rows, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16); X = X.astype(np.float64)
tr, _ = S.pool_split(valid, 0); mdl = S.model(0).fit(X[tr] @ T.T, y[tr]); w = T.T @ (mdl[-1].coef_[0] / mdl[0].scale_)
dmu = X[y == 1].mean(0) - X[y == 0].mean(0)
_, _, Vp = np.linalg.svd(T - T.mean(0), full_matrices=False)
def share(V, v, k):
    Qk, _ = np.linalg.qr(V[:k].T); return float(np.sum((Qk.T @ v) ** 2) / np.sum(v ** 2))
KS = [5, 10, 20, 48, 96, 228]; out = {}
lines = ["| axes | persona-ID acc k=5 / 10 / 20 | w share k=10 / 50 / 228 | Δμ share k=10 / 228 | xfer k=5 | k=10 | k=20 | k=48 | k=96 | k=228 |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
for name, conf, lab, V in [("pca", P, "a1(pca)", Vp[:228])] + [(b, L, b[4:], B[b]) for b in ["lda_a0.3", "lda_a0.1", "lda_a0.01", "lda_lw"]]:
    acc = chk[lab]["acc"]; cells = []
    for k in KS:
        key = f"{name}_k{k}"; hb, xf, ci = valC(lambda c: conf[c][key]["per_seed"]); cells.append(f"{xf:.3f}"); out[key] = dict(hb=hb, xfer=xf, ci=ci)
    ws = [share(V, w, k) for k in (10, 50, 228)]; ds = [share(V, dmu, k) for k in (10, 228)]
    lines.append(f"| {name} (α={chk[lab]['alpha']:.3g}) | {acc['5']:.2f} / {acc['10']:.2f} / {acc['20']:.2f} | " + " / ".join(f"{x:.2f}" for x in ws) + " | " + " / ".join(f"{x:.2f}" for x in ds) + " | " + " | ".join(cells) + " |")
    out[name + "_shares"] = dict(w=ws, dmu=ds)
pj = valC(lambda c: P[c]["traits_k229"]["per_seed"])
lines.append(f"\nPersonaJB (all traits), C on val: HB {pj[0]:.3f}, xfer {pj[1]:.3f} ± {pj[2]:.3f}. Δμ share of span(T) = {share(T, dmu, 229):.2f}.")
print("\n".join(lines)); open("tables_lda.md", "w").write("\n".join(lines)); json.dump(out, open("aggregate_lda.json", "w"), indent=1)
