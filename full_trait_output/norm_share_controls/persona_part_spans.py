#!/usr/bin/env python3
"""
Per-vector split test of persona_part_pervector.py, repeated along other 229-dim spans: does the
"inside part carries the transfer signal, the rest is noise" picture hold only for the trait span?
Vectors: the 1145 Wikipedia / No Robots / Alpaca activations (5 draws of 229) of persona_part_pervector.py.
Spans used for the split v = v_in + v_out:
  persona     the 229 trait vectors
  No Robots   229 No Robots activations from prompts used in none of the 5 No Robots draws (held out)
  Wikipedia   229 Wikipedia activations from passages used in none of the 5 Wikipedia draws (held out)
  random      229 random Gaussian directions (gauss_d0)
Transfer AUC of h.v, h.v_in, h.v_out (folded, mean of the 5 families); counts over the top 50 vectors by HarmBench
AUC; noise bar = 95th percentile of 1145 random Gaussian directions.
Writes persona_part_spans.json, figs/fig_persona_part_spans.pdf/png (rows = span, columns = vector source).
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
HERE = Path(__file__).resolve().parent
exec(compile(open(HERE / "persona_part_pervector.py").read().split("TOP = 50")[0], str(HERE / "persona_part_pervector.py"), "exec"))

def held_out(path, reading, src):  # rebuild build_steer_wiki.py's draws, then pick 229 unused prompts
    D = torch.load(HERE / path, map_location="cpu"); ids = sorted(D[reading], key=lambda s: int(s.split("_")[1]))
    M = np.stack([D[reading][i].float().numpy() for i in ids]).astype(np.float64)
    r = np.random.default_rng(1); picks = [r.choice(len(M), 229, replace=False) for _ in range(5)]
    for d, p in enumerate(picks): assert np.allclose(unit_rows(M[p]), eb[f"{src}_d{d}"], atol=1e-4), (src, d)
    free = np.setdiff1d(np.arange(len(M)), np.concatenate(picks))
    return unit_rows(M[np.random.default_rng(7).choice(free, 229, replace=False)]), len(free)
NRh, nf = held_out("wiki/norobots_acts.pt", "chat", "nr"); WKh, wf = held_out("wiki/wiki_acts.pt", "raw", "wiki")
print(f"held-out pools: No Robots {nf}, Wikipedia {wf}")
SP = {"persona": T, "No Robots (held out)": NRh, "Wikipedia (held out)": WKh, "random": cb["gauss_d0"].astype(np.float64)}
TOP = 50; res = {"p95": p95}
fig, axs = plt.subplots(len(SP), 3, figsize=(397 / 72, 6.9), sharex=True, sharey=True)
print(f"{'split along':22s} {'vectors':10s} | share in | transfer: v  v_in  v_out | top-50: v_in>=v  v_out<=noise")
for r, (sn, B) in enumerate(SP.items()):
    Q, _ = np.linalg.qr(B.T)
    for c, (dn, V) in enumerate(DICTS.items()):
        V = unit_rows(V.astype(np.float64)); top = np.argsort(-hbA(V))[:TOP]
        Vin = V @ Q @ Q.T; x, xi, xo = xfA(V), xfA(unit_rows(Vin)), xfA(unit_rows(V - Vin))
        sh = float(np.median(np.sum(Vin ** 2, 1)))
        res[f"{sn}|{dn}"] = dict(share=sh, xf=x[top].mean(), xf_in=xi[top].mean(), xf_out=xo[top].mean(),
                                 n_in_ge=int(np.sum(xi[top] >= x[top])), n_out_le=int(np.sum(xo[top] <= p95)))
        q = res[f"{sn}|{dn}"]
        print(f"{sn:22s} {dn:10s} | {sh:8.3f} | {q['xf']:10.3f} {q['xf_in']:5.3f} {q['xf_out']:6.3f} | {q['n_in_ge']:8d}/50 {q['n_out_le']:8d}/50")
        ax = axs[r, c]
        ax.scatter(x, xo, s=2, c="#B0B0B0", lw=0, label=r"remainder $v_{\mathrm{out}}$")
        ax.scatter(x, xi, s=2, c="#3B6FB6", lw=0, label=r"component inside the span $v_{\mathrm{in}}$")
        ax.scatter(x[top], xi[top], s=7, facecolors="none", edgecolors="#1B3F7A", lw=0.6, label=f"top {TOP} by HarmBench AUC")
        ax.plot([0.5, 0.78], [0.5, 0.78], c="k", lw=0.5); ax.axhline(p95, c="#C04040", lw=0.6, ls="--")
        ax.set_xlim(0.5, 0.78); ax.set_ylim(0.5, 0.78); ax.tick_params(labelsize=6, length=2)
        if r == 0: ax.set_title(f"{dn} activations", fontsize=7.8)
        if r == len(SP) - 1: ax.set_xlabel("transfer AUC of $v$", fontsize=7)
        if c == 0: ax.set_ylabel(f"split along\n{sn} span", fontsize=7)
        t = ax.text(0.51, 0.765, f"share in span {sh:.2f}", fontsize=5.8, color="#555555", va="top")
        t.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=6.8, frameon=False, markerscale=2.5)
fig.tight_layout(rect=(0, 0.03, 1, 1), pad=0.3, h_pad=0.5, w_pad=0.5)
for e in ("pdf", "png"): fig.savefig(HERE / f"figs/fig_persona_part_spans.{e}", dpi=200)
json.dump(res, open(HERE / "persona_part_spans.json", "w"), indent=1)
