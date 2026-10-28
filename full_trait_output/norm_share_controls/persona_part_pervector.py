#!/usr/bin/env python3
"""
Per-vector version of the persona-part test (no medians over unpredictive vectors).
Vectors: all 5x229 = 1145 unit activations per source (Wikipedia, No Robots, Alpaca). For each vector v, split
along a 229-dim span: v_in = shadow of v on the span, v_out = v - v_in. Score every prompt by h.v, h.v_in, h.v_out;
folded AUC on HarmBench and mean folded AUC over the 5 transfer families.
The predictive vectors are picked by their HarmBench AUC (independent of the transfer labels): top 50 per source.
Floor: same transfer AUC for 1145 random Gaussian directions (mean, 95th percentile).
Writes persona_part_pervector.json and figs/fig_persona_part_pervector.pdf/png (persona split, all vectors).
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
exec(compile(open(HERE / "persona_part_test.py").read().split("res = {}")[0], str(HERE / "persona_part_test.py"), "exec"))
DICTS = {"Wikipedia": np.concatenate([eb[f"wiki_d{i}"] for i in range(5)]),
         "No Robots": np.concatenate([eb[f"nr_d{i}"] for i in range(5)]),
         "Alpaca": np.concatenate([cb[f"acts_d{i}"] for i in range(5)])}
hbA = lambda V: fold(rank_auc(yh, Xh @ V.T))
xfA = lambda V: np.mean([fold(rank_auc(y, X @ V.T)) for X, y in fam.values()], axis=0)
G = unit_rows(np.random.default_rng(5).standard_normal((1145, 4096)))
g = xfA(G); floor, p95 = float(g.mean()), float(np.quantile(g, .95))
print(f"random-direction transfer floor: mean {floor:.3f}, 95th pct {p95:.3f}")
TOP = 50
res, fig, axs = {"floor": floor, "p95": p95}, *plt.subplots(1, 3, figsize=(6.8, 2.55), sharex=True, sharey=True)
for ax, (dn, V) in zip(axs, DICTS.items()):
    V = unit_rows(V.astype(np.float64)); top = np.argsort(-hbA(V))[:TOP]
    x_v = xfA(V)
    print(f"\n=== {dn}: top {TOP} of {len(V)} vectors by HarmBench AUC (HB {hbA(V)[top].mean():.3f})")
    print(f"{'split along':20s} | transfer: whole v   v_in   v_out | #vectors v_in >= v   #vectors v_out <= floor p95")
    for sn, B in SPANS.items():
        Q, _ = np.linalg.qr(B.T); Vin = V @ Q @ Q.T; Vout = V - Vin
        x_in, x_out = xfA(unit_rows(Vin)), xfA(unit_rows(Vout))
        t = top
        print(f"{sn:20s} | {x_v[t].mean():16.3f} {x_in[t].mean():6.3f} {x_out[t].mean():7.3f} | "
              f"{int(np.sum(x_in[t] >= x_v[t])):10d}/{TOP} {int(np.sum(x_out[t] <= p95)):20d}/{TOP}")
        res[f"{dn}|{sn}"] = {"top_idx": top.tolist(), "xf_v": x_v.tolist(), "xf_in": x_in.tolist(), "xf_out": x_out.tolist()}
        if sn == "Persona traits":
            ax.scatter(x_v, x_out, s=2.5, c="#B0B0B0", label=r"remainder $v_\perp$", lw=0)
            ax.scatter(x_v, x_in, s=2.5, c="#3B6FB6", label=r"persona component $v_\parallel$", lw=0)
            ax.scatter(x_v[t], x_in[t], s=9, facecolors="none", edgecolors="#1B3F7A", lw=0.7, label=f"top {TOP} by HarmBench AUC")
    lo, hi = 0.5, 0.78
    ax.plot([lo, hi], [lo, hi], c="k", lw=0.6); ax.axhline(p95, c="#C04040", lw=0.7, ls="--")
    ax.set_title(f"{dn} activations", fontsize=8); ax.set_xlabel("transfer AUC of $v$", fontsize=7.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.tick_params(labelsize=6.5, length=2)
axs[0].set_ylabel("transfer AUC of the component", fontsize=7.5)
import matplotlib.patheffects as pe
axs[0].text(0.505, p95 + 0.005, "random directions, 95th pct", color="#C04040", fontsize=6, zorder=5,
             path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=7, frameon=False, markerscale=2.5)
fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.3, w_pad=0.6); (HERE / "figs").mkdir(exist_ok=True)
for e in ("pdf", "png"): fig.savefig(HERE / f"figs/fig_persona_part_pervector.{e}", dpi=200)
json.dump(res, open(HERE / "persona_part_pervector.json", "w"))
