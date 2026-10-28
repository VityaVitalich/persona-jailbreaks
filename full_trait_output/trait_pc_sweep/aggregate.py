#!/usr/bin/env python3
"""Tables + figure for the trait-PC sweep. Reads summary_<model>{,_C0.01,_C0.1,_C1}.json (C=10 is the untagged run).
'val-C' = per seed, the C in {0.01,0.1,1,10} with the best HarmBench inner-val AUC; its test/transfer AUCs are used.
'val-k,C' = per seed, (k, C) jointly chosen on the same val AUC within a basis."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = Path(__file__).parent; FAMS = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
CS = {"0.01": "_C0.01", "0.1": "_C0.1", "1": "_C1", "10": ""}
out = {}; md = []
def ci(x): return 1.96*np.std(x, ddof=1)/np.sqrt(len(x))
for m in ["llama", "gemma"]:
    S = {c: json.load(open(H/f"summary_{m}{t}.json"))["configs"] for c, t in CS.items()}
    pub = json.load(open(H/f"summary_{m}.json"))["published_personajb"]
    out[m] = {}
    for b in ["pca", "svd", "traits"]:
        names = [n for n in S["10"] if S["10"][n]["basis"] == b and all(n in S[c] for c in CS)]
        rows = []
        for n in names:
            ps = {c: S[c][n]["per_seed"] for c in CS}
            val = np.array([ps[c]["val"]["auc"] for c in CS])                 # [C, seed]
            pick = np.nanargmax(val, 0); Cl = list(CS)
            per = {f: np.array([ps[Cl[pick[s]]][f]["auc"][s] for s in range(val.shape[1])]) for f in ["human_test"] + FAMS}
            xf = np.mean([per[f] for f in FAMS], 0)
            fx = S["10"][n]
            rows.append(dict(k=fx["k"], ev=fx["trait_var_explained"],
                fixed=dict(hb=fx["mean"]["human_test"], xfer=fx["avg_transfer"], ci=fx["avg_transfer_ci95"], fam={f: fx["mean"][f] for f in FAMS}, signed=fx["avg_transfer_signed"]),
                valC=dict(hb=float(np.nanmean(per["human_test"])), xfer=float(xf.mean()), ci=float(ci(xf)), fam={f: float(per[f].mean()) for f in FAMS},
                          C_mode=Cl[np.bincount(pick, minlength=4).argmax()], C_counts={Cl[i]: int((pick == i).sum()) for i in range(4)}),
                _val=val, _ps=ps))
        rows.sort(key=lambda r: r["k"])
        # joint (k, C) selection on val, per seed
        if b != "traits":
            V = np.stack([r["_val"] for r in rows])  # [k, C, seed]
            nS = V.shape[2]; sel = []
            for s in range(nS):
                ki, ci_ = np.unravel_index(np.nanargmax(V[:, :, s]), V.shape[:2]); sel.append((ki, ci_))
            Cl = list(CS)
            xf = np.array([np.mean([rows[ki]["_ps"][Cl[cj]][f]["auc"][s] for f in FAMS]) for s, (ki, cj) in enumerate(sel)])
            hb = np.array([rows[ki]["_ps"][Cl[cj]]["human_test"]["auc"][s] for s, (ki, cj) in enumerate(sel)])
            ks = [rows[ki]["k"] for ki, _ in sel]
            out[m][b + "_valkC"] = dict(xfer=float(xf.mean()), ci=float(ci(xf)), hb=float(np.nanmean(hb)), k_median=float(np.median(ks)), k_range=[int(min(ks)), int(max(ks))])
        for r in rows: r.pop("_val"); r.pop("_ps")
        out[m][b] = rows
    out[m]["published"] = pub
json.dump(out, open(H/"aggregate.json", "w"), indent=1)

for m, title in [("llama", "Llama-3.1-8B, layer 16, 229 traits"), ("gemma", "Gemma-4-31B, layer 30, 240 traits")]:
    O = out[m]; t = O["traits"][0]
    md += [f"### {title}", "", f"PersonaJB (all traits, C=10, published): HB {t['fixed']['hb']:.3f}, avg transfer {t['fixed']['xfer']:.3f} ± {t['fixed']['ci']:.3f}. "
           f"Same with val-selected C: {t['valC']['hb']:.3f} / {t['valC']['xfer']:.3f} ± {t['valC']['ci']:.3f}.", "",
           "Centered trait PCs (`pca`). Fixed C=10 = PersonaJB's config; val-C = C picked per seed on HarmBench inner val.", "",
           "| k | trait var. | HB test (C=10) | Avg. transfer (C=10) | HB test (val-C) | GCG | PAIR | PAP | GPTFuzz | PEZ | Avg. transfer (val-C) | modal C |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in O["pca"]:
        f, v = r["fixed"], r["valC"]
        md.append(f"| {r['k']} | {r['ev']:.2f} | {f['hb']:.3f} | {f['xfer']:.3f} ± {f['ci']:.3f} | {v['hb']:.3f} | " + " | ".join(f"{v['fam'][x]:.3f}" for x in FAMS) + f" | {v['xfer']:.3f} ± {v['ci']:.3f} | {v['C_mode']} |")
    md += [f"| **PersonaJB** | 1.00 | {t['fixed']['hb']:.3f} | {t['fixed']['xfer']:.3f} ± {t['fixed']['ci']:.3f} | {t['valC']['hb']:.3f} | " + " | ".join(f"{t['valC']['fam'][x]:.3f}" for x in FAMS) + f" | {t['valC']['xfer']:.3f} ± {t['valC']['ci']:.3f} | {t['valC']['C_mode']} |", ""]
    for b in ["pca", "svd"]:
        j = O[b + "_valkC"]
        md.append(f"- `{b}` with k and C both chosen on val per seed: HB {j['hb']:.3f}, avg transfer {j['xfer']:.3f} ± {j['ci']:.3f} (median k {j['k_median']:.0f}, range {j['k_range']}).")
    md.append("")
(H/"tables.md").write_text("\n".join(md)); print("\n".join(md))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (m, title) in zip(axes, [("llama", "Llama-3.1-8B (229 traits)"), ("gemma", "Gemma-4-31B (240 traits)")]):
    O = out[m]
    for key, lab, col, ls in [("fixed", "trait PCs, C=10", "#999999", "--"), ("valC", "trait PCs, C on val", "#0F766E", "-")]:
        k = [r["k"] for r in O["pca"]]; y = np.array([r[key]["xfer"] for r in O["pca"]]); e = np.array([r[key]["ci"] for r in O["pca"]])
        ax.plot(k, y, ls, color=col, marker="o", ms=3, label=lab); ax.fill_between(k, y-e, y+e, color=col, alpha=.15, lw=0)
    t = O["traits"][0]["fixed"]
    ax.axhline(t["xfer"], color="#B8860B", lw=1.2); ax.axhspan(t["xfer"]-t["ci"], t["xfer"]+t["ci"], color="#B8860B", alpha=.15, lw=0)
    ax.text(1.05, t["xfer"]+.004, "PersonaJB (all traits)", color="#B8860B", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("number of trait-space PCs k"); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
axes[0].set_ylabel("avg transfer AUC (5 unseen families)"); axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(H/"fig_trait_pc_sweep.pdf"); fig.savefig(H/"fig_trait_pc_sweep.png", dpi=150)
