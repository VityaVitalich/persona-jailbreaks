#!/usr/bin/env python3
"""
plot_alpha_coherent_asr.py

ASR-among-coherent = jailbroken / (n - degenerate) vs steering alpha, for the
three real directions (PersonaJB / Refusal / DMU). Reads the per-alpha,
non-escalating sweep (outputs_alpha_sweep/a<alpha>/harmbench/<arm>/results.jsonl),
where every one of the 300 refused pairs was steered at exactly that alpha.

  python full_trait_tools/plot_alpha_coherent_asr.py \
      --root full_trait_output/outputs_alpha_sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ARMS = [                                   # (dir key, label, color)
    ("w_persona__full",        "PersonaJB", "#1b3a6b"),
    ("refusal_pos1_neg__full", "Refusal",   "#2e8b8b"),
    ("dmu__full",              "DMU",       "#c46a1b"),
]


def cell(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(rows)
    jb = sum(1 for r in rows if r.get("final_jailbroken"))
    deg = sum(1 for r in rows if r.get("final_degenerate"))
    coh = n - deg
    return dict(n=n, jb=jb, deg=deg, coherent=coh,
                asr_coherent=(jb / coh if coh else float("nan")),
                asr_raw=jb / n if n else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "full_trait_output/outputs_alpha_sweep"))
    ap.add_argument("--out", default=str(ROOT / "full_trait_output/plots/alpha_coherent_asr"))
    args = ap.parse_args()
    root = Path(args.root)

    alphas = sorted(float(d.name[1:]) for d in root.glob("a*") if d.is_dir())
    print(f"alphas: {alphas}\n")

    series = {}
    for key, label, _ in ARMS:
        ys, meta = [], []
        for a in alphas:
            p = root / f"a{a}" / "harmbench" / key / "results.jsonl"
            c = cell(p)
            ys.append(c["asr_coherent"])
            meta.append(c)
        series[key] = (label, ys, meta)
        print(f"{label}")
        for a, m in zip(alphas, meta):
            print(f"  a={a}: jb={m['jb']:3d} deg={m['deg']:3d} coherent={m['coherent']:3d} "
                  f"-> jb/(n-deg)={m['asr_coherent']:.3f}  (raw jb/n={m['asr_raw']:.3f})")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, label, color in ARMS:
        _, ys, _ = series[key]
        ax.plot(alphas, ys, "-o", color=color, lw=2, ms=7, label=label)
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel("ASR among coherent outputs\n" r"jailbroken / (n $-$ degenerate)")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(alphas)
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"\nwrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
