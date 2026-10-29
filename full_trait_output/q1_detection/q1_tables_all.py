"""Cross-model markdown tables from out/q1_{model}_{train}.json and out/q1_posthoc_{model}_{train}.json:
(1) avg folded transfer AUC (PAP/GCG/PAIR/GPTFuzz/PEZ) per detector x model, per training set;
(2) delta vs full when the persona component is removed before training (same readout, C re-tuned) and after training,
with the random-span null alongside. Missing files print "pending".   usage: python q1_tables_all.py"""
import json, os
MODELS = ["llama", "olmo", "qwen", "gemma"]; TRAINS = {"hb": "HB", "hbwjb": "HB+WJB"}
NAMES = {"PersonaJB": "PersonaJB", "raw": "Linear probe (raw)", "PCA": "PCA-K (train acts)", "gauss": "Random Gaussian K (3 draws; post-hoc CI = mean of draw CIs)",
         "alpaca_pca": "Alpaca subspace (PCA-K)", "wiki_pca": "Wikipedia subspace (PCA-K)", "jb_dir": "JB direction (Δμ)", "refusal": "Refusal direction",
         "alpaca_mean": "Alpaca mean direction", "wiki_mean": "Wikipedia mean direction"}
load = lambda f: json.load(open(f)) if os.path.exists(f) else None
ci = lambda a, lo, hi: f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
dci = lambda d, lo, hi: f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
for tr, trn in TRAINS.items():
    Q = {m: load(f"out/q1_{m}_{tr}.json") for m in MODELS}; P = {m: load(f"out/q1_posthoc_{m}_{tr}.json") for m in MODELS}
    print(f"\n### Trained on {trn}: avg transfer AUC (PAP, GCG, PAIR, GPTFuzz, PEZ), 95% CI" + (" + WJB transfer" if tr == "hb" else ""))
    print("| Detector | " + " | ".join(MODELS) + " |\n|" + "---|" * (len(MODELS) + 1))
    for d, n in NAMES.items():
        cells = []
        for m in MODELS:
            if Q[m] is None: cells.append("pending"); continue
            s = Q[m]["selected"][f"{d}__full__summary"]; c = ci(s["avg"], *s["avg_ci95"])
            if tr == "hb":
                ks = [f"gauss_d{k}__full__reselect" for k in range(3)] if d == "gauss" else [f"{d}__full__reselect"]
                c += f" / {sum(Q[m]['selected'][k]['test_auc']['wjb'] for k in ks) / len(ks):.3f}"
            cells.append(c)
        print(f"| {n} | " + " | ".join(cells) + " |")
    for title, src in (("removed BEFORE training (retrain, same readout)", "pre"), ("removed AFTER training (trained detector, persona-removed test acts)", "post")):
        print(f"\n### Trained on {trn}: Δ avg transfer, persona {title}; [random-span null]")
        print("| Detector | " + " | ".join(MODELS) + " |\n|" + "---|" * (len(MODELS) + 1))
        for d, n in NAMES.items():
            cells = []
            for m in MODELS:
                if src == "pre":
                    if Q[m] is None: cells.append("pending"); continue
                    a, b = Q[m]["delta"].get(f"{d}__no_persona__fixed"), Q[m]["delta"].get(f"{d}__no_random__fixed")
                    cells.append("— (by construction)" if a is None else f"{dci(a['delta'], *a['ci95'])} [{b['delta']:+.3f}]")
                else:
                    if P[m] is None: cells.append("pending"); continue
                    ks = [f"gauss_d{k}" for k in range(3)] if d == "gauss" else [d]; o = [P[m]["detectors"][k] for k in ks]
                    a = sum(x["no_persona"]["delta"] for x in o) / len(o); b = sum(x["no_random"]["delta"] for x in o) / len(o)
                    lo = sum(x["no_persona"]["delta_ci95"][0] for x in o) / len(o); hi = sum(x["no_persona"]["delta_ci95"][1] for x in o) / len(o)
                    w = sum(x["w_share_persona"] for x in o) / len(o)
                    cells.append(f"{dci(a, lo, hi)} [{b:+.3f}] w∈P {w:.2f}")
            print(f"| {n} | " + " | ".join(cells) + " |")
