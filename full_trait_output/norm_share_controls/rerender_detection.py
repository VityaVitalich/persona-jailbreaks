#!/usr/bin/env python3
"""
Local re-render of fig_detection_transfer_transposed and fig_id_vs_transfer with control rows: the PersonaJB
detector (same pipeline, C=10, 50 pool splits) on 229 random activations of Wikipedia / No Robots / Alpaca,
229 shuffled-label Alpaca contrasts, and 229 random Gaussian directions (Llama only; no Gemma bar).
Values are the paper metric (folded AUC), mean over 5 draws; whisker = half the min-max range over draws
(larger than the split CI). Read from trait_pc_sweep/summary_llama_{subspace,extra}_controls.json.
The generator full_trait_tools/plot_paper_result_figs.py is used unchanged apart from the patches below;
output goes to norm_share_controls/figs/.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = REPO / "full_trait_tools/plot_paper_result_figs.py"
SW = REPO / "full_trait_output/trait_pc_sweep"
C = {**json.load(open(SW / "summary_llama_subspace_controls.json"))["configs"],
     **json.load(open(SW / "summary_llama_extra_controls.json"))["configs"]}
assert abs(C["traits_k229"]["mean"]["human_test"] - 0.854) < 0.001 and abs(C["traits_k229"]["avg_transfer"] - 0.760) < 0.001

ROWS = [("Wikipedia (229)", "wiki_d"), ("No Robots (229)", "nr_d"), ("Alpaca (229)", "acts_d"),
        ("Shuffled (229)", "noise_d"), ("Random (229)", "gauss_d")]
ctrl = []
for name, pre in ROWS:
    hb = np.array([C[f"{pre}{i}_k229"]["mean"]["human_test"] for i in range(5)])
    xf = np.array([C[f"{pre}{i}_k229"]["avg_transfer"] for i in range(5)])
    ctrl.append((name, round(float(hb.mean()), 3), round(float(xf.mean()), 3), round(float((xf.max() - xf.min()) / 2), 3), "ctrl"))
    print(f"  {name:26s} HB {hb.mean():.3f}  transfer {xf.mean():.3f}  [{xf.min():.3f}, {xf.max():.3f}]")

src = GEN.read_text()
src = src.replace("    fig, ax = new_ax((5.5, 1.95))", "    fig, ax = new_ax((7.4, 1.95))")
src = src.replace('CLASS_COLOR = {"ours": C_OURS, "raw": C_RAW, "base": C_BASE}',
                  'CLASS_COLOR = {"ours": C_OURS, "raw": C_RAW, "base": C_BASE, "ctrl": "#7A5195"}')
src = src.replace("        ge_v, ge_ci = DET_GEMMA[name]", "        ge_v, ge_ci = DET_GEMMA.get(name, (float('nan'), 0.0))")
src = src.replace("        dx, dy, ha = offs[name]", "        dx, dy, ha = offs.get(name, CTRL_OFFS.get(name, (7, -3, 'left')))")
src = src.replace("[{\"PersonaJB Direction\": \"PersonaJB\\nDirection\",",
                  "[{\"PersonaJB Direction\": \"PersonaJB\\nDirection\", **{n: '\\n'.join(n.rsplit(' ', 1)) for n in CTRL_NAMES},")
src = src.replace("               color=C_LLAMA, zorder=3)", "               color=CLASS_COLOR['ctrl'] if cls == 'ctrl' else C_LLAMA, zorder=3)")
src = src.replace('''                    loc="upper right", ncol=1,''', '''                    loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,''')
src = src.replace('''    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C_LLAMA, C_GEMMA)]
    leg = ax.legend(handles, ["Llama-3.1-8B", "Gemma-4-31B"],''', '''    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C_LLAMA, C_GEMMA, CLASS_COLOR["ctrl"])]
    leg = ax.legend(handles, ["Llama-3.1-8B", "Gemma-4-31B", "PersonaJB detector on 229 non-persona directions (Llama)"],''')
src = src.replace('''                        xytext=(dx, dy), ha=ha, fontsize=7.5,''', '''                        xytext=(dx, dy), ha=ha, fontsize=7.5, va="center" if cls == "ctrl" else "baseline",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.7, shrinkA=1, shrinkB=3) if cls == "ctrl" and abs(dy) > 12 else None,''')
src = src.replace('''    save(fig, outdir, "fig_id_vs_transfer", png)''', '''    ax.text(0.985, 0.49, "purple: PersonaJB detector on\\n229 non-persona directions\\n(Llama, mean of 5 draws)", ha="right", va="bottom", fontsize=7, color=CLASS_COLOR["ctrl"])
    save(fig, outdir, "fig_id_vs_transfer", png)''')
src = src.replace('if __name__ == "__main__":\n    main()', "")
ns = {"__name__": "rerender", "__file__": str(GEN)}
exec(compile(src, str(GEN), "exec"), ns)
ns["DET"].extend(ctrl)
ns["CTRL_NAMES"] = [r[0] for r in ROWS]
ns["CTRL_OFFS"] = {"Alpaca (229)": (-76, 51, "right"), "No Robots (229)": (-70, 49 - 10, "right"),
                   "Wikipedia (229)": (-73, 39 - 10, "right"), "Shuffled (229)": (-52, 33 - 10, "right"),
                   "Random (229)": (7, -1, "left")}
out = HERE / "figs"; out.mkdir(exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
ns["fig_detection_transfer_transposed"](out, True)
ns["fig_id_vs_transfer"](out, True)
for n in ("fig_detection_transfer_transposed", "fig_id_vs_transfer"):
    (out / f"{n}.pdf").rename(out / f"{n}_controls.pdf"); (out / f"{n}.png").rename(out / f"{n}_controls.png")
