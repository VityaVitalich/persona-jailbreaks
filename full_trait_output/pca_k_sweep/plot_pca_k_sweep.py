#!/usr/bin/env python3
"""Plot AUC vs PCA dimensionality k for both models (paper cell protocol), plus a markdown table."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

S = Path(__file__).parent  # script + sweep_{llama,gemma}.json live together
FAMS = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
# categorical slots, fixed order (reference palette, light): blue, orange, aqua, yellow, magenta, green, violet
C = {"avg": "#2a78d6", "human_test": "#52514e", "GCG": "#eb6834", "PAIR": "#1baf7a", "PAP": "#eda100", "GPTFuzz": "#e87ba4", "PEZ": "#4a3aa7"}
PAPER_K   = {"llama": 4, "gemma": 3}
PERSONAJB = {"llama": 0.76, "gemma": 0.74}   # Avg. Transfer of the PersonaJB Direction row
TITLE     = {"llama": "Llama-3.1-8B (layer 16)", "gemma": "Gemma-4-31B (layer 30)"}

def load(model):
    d = json.load(open(S / f"sweep_{model}.json"))
    ks = sorted(int(k) for k in d["per_k"]); rows = []
    for k in ks:
        pf = d["per_k"][str(k)]["per_family"]
        xfer = np.nanmean(np.array([pf[f]["auc_paper"] for f in FAMS]), axis=0)  # per-seed avg transfer
        xfer_signed = np.nanmean(np.array([pf[f]["auc"] for f in FAMS]), axis=0)
        rows.append(dict(k=k, ev=d["per_k"][str(k)]["explained_var"],
                         hb=np.nanmean(pf["human_test"]["auc_paper"]),
                         fam={f: np.nanmean(pf[f]["auc_paper"]) for f in FAMS},
                         fam_signed={f: np.nanmean(pf[f]["auc"]) for f in FAMS},
                         xfer=xfer.mean(), xfer_ci=1.96*xfer.std(ddof=1)/np.sqrt(len(xfer)),
                         xfer_signed=xfer_signed.mean(), n=len(xfer)))
    return d, rows

def main():
    models = [m for m in ("llama", "gemma") if (S / f"sweep_{m}.json").exists()]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2*len(models), 3.9), squeeze=False)
    md = []
    for ax, m in zip(axes[0], models):
        d, rows = load(m)
        pca_rows = [r for r in rows if r["k"] > 0]; raw = next((r for r in rows if r["k"] == 0), None)
        ks = [r["k"] for r in pca_rows]
        xmax = max(ks) * 2.2
        for f in FAMS:
            ax.plot(ks, [r["fam"][f] for r in pca_rows], color=C[f], lw=1.2, alpha=0.85, marker="o", ms=3, label=f)
        ax.plot(ks, [r["hb"] for r in pca_rows], color=C["human_test"], lw=1.2, ls=":", marker="o", ms=3, label="HarmBench test")
        xf = np.array([r["xfer"] for r in pca_rows]); ci = np.array([r["xfer_ci"] for r in pca_rows])
        ax.fill_between(ks, xf-ci, xf+ci, color=C["avg"], alpha=0.15, lw=0)
        ax.plot(ks, xf, color=C["avg"], lw=2.4, marker="o", ms=4.5, label="Avg. transfer (5 families)")
        if raw is not None:   # raw activations reference at the right edge
            ax.scatter([xmax], [raw["xfer"]], color=C["avg"], s=36, zorder=5, marker="s")
            ax.annotate("raw\n(no PCA)", (xmax, raw["xfer"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="#52514e")
        ax.axhline(PERSONAJB[m], color="#0b0b0b", lw=1, ls="--")
        ax.text(ks[0], PERSONAJB[m]+0.006, "PersonaJB Direction (paper)", fontsize=7.5, color="#0b0b0b", va="bottom")
        pk = PAPER_K[m]; pr = next(r for r in pca_rows if r["k"] == pk)
        ax.axvline(pk, color="#52514e", lw=0.8, ls="-.")
        ax.annotate(f"paper: k={pk}\navg xfer {pr['xfer']:.2f}", (pk, 0.365), fontsize=7.5, ha="left", color="#52514e", xytext=(4, 0), textcoords="offset points")
        best = max(pca_rows, key=lambda r: r["xfer"])
        ax.annotate(f"best k={best['k']}: {best['xfer']:.2f}", (best["k"], best["xfer"]), xytext=(best["k"], 0.455), textcoords="data", ha="center", fontsize=7.5, color=C["avg"], fontweight="bold", arrowprops=dict(arrowstyle="-", color=C["avg"], lw=0.8, shrinkB=4))
        ax.set_xscale("log", base=2); maj = [k for k in ks if k & (k-1) == 0]; ax.set_xticks(maj); ax.set_xticklabels([str(k) for k in maj], fontsize=7.5)
        ax.set_xticks([k for k in ks if k not in maj], minor=True); ax.tick_params(axis="x", which="minor", length=2); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(ks[0]/1.4, xmax*1.6); ax.set_ylim(0.35, 0.95)
        ax.set_xlabel("PCA components k (HarmBench basis)"); ax.set_ylabel("AUC (paper cell protocol)")
        ax.set_title(TITLE[m], fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
        # markdown table
        md.append(f"\n### {TITLE[m]} — {d['protocol']}, {rows[0]['n']} seeds\n")
        md.append("| k | expl. var | HB test | " + " | ".join(FAMS) + " | Avg. transfer ±95% CI | Avg. transfer (signed AUC) |")
        md.append("|---:|---:|---:|" + "---:|"*len(FAMS) + "---:|---:|")
        for r in rows:
            lab = "raw" if r["k"] == 0 else str(r["k"])
            flag = " ← paper" if r["k"] == PAPER_K[m] else (" ← best" if r is best else "")
            md.append(f"| {lab}{flag} | {r['ev']:.2f} | {r['hb']:.3f} | " + " | ".join(f"{r['fam'][f]:.3f}" for f in FAMS) + f" | {r['xfer']:.3f} ± {r['xfer_ci']:.3f} | {r['xfer_signed']:.3f} |")
    h, l = axes[0][0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(S / f"fig_pca_k_sweep.{ext}", dpi=200, bbox_inches="tight")
    (S / "pca_k_sweep_tables.md").write_text("\n".join(md))
    print("\n".join(md)); print("saved", S / "fig_pca_k_sweep.pdf")

if __name__ == "__main__":
    main()
