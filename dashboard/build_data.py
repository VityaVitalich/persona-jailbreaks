#!/usr/bin/env python3
"""Collect the persona-jailbreaks result files already saved on disk into one data.json for the static dashboard.

Nothing is recomputed: every number is read from a saved result file. The only arithmetic is re-aggregation of
saved numbers (mean over saved per-seed / per-draw / per-family AUCs, a rate over saved per-prompt labels, or the
difference of two saved point estimates). AUCs are always folded, max(a, 1 - a): files that also store a signed AUC
have that field dropped, and any AUC read from a file whose convention is folded is re-folded defensively.

usage: python3 build_data.py [--smoke-q1 PATH]   (--smoke-q1 is for development only: it loads a q1 smoke output
        and flags it as SMOKE in the page; the default build never reads smoke files)
"""
import argparse, csv, glob, gzip, json, math, os, re, sys, time
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent
# Inside the public repo (dashboard/ next to full_trait_output/) read the repo; otherwise the local research checkout.
PVJ = Path(os.environ["PJB_ROOT"]) if os.environ.get("PJB_ROOT") else (HERE.parent if (HERE.parent / "full_trait_output").is_dir() else HOME / "persona-vector-jbing")
RELROOT = PVJ / "data/release" if (PVJ / "data/release").is_dir() else HOME / "pjb_release_staging"
FTO = PVJ / "full_trait_output"
DET = FTO / "olmo3_7b/detect"
NSC = FTO / "norm_share_controls"

MODELS = {"llama": "Llama-3.1-8B", "gemma": "Gemma-4-31B", "olmo": "OLMo-3-7B", "qwen": "Qwen3-32B"}
SETS = {"hb_test": "HB test", "hb": "HB", "wjb": "WJB", "pap": "PAP", "gcg": "GCG", "pair": "PAIR", "gptfuzz": "GPTFuzz",
        "pez": "PEZ", "hb_randomCV": "HB random CV", "hb_disjoint": "HB disjoint test"}
FAM5 = ["pap", "gcg", "pair", "gptfuzz", "pez"]

# ---------------------------------------------------------------- detector catalogue (id -> label, group)
DETS = {
    "PersonaJB": ("PersonaJB", "persona"),
    "PersonaJB_filtered": ("PersonaJB (228 Llama-filtered traits)", "persona"),
    "raw": ("Raw linear probe", "baseline"),
    "PCA": ("PCA (same size)", "baseline"),
    "gauss": ("Random Gaussian projections (mean of draws)", "baseline"),
    "jb_dir": ("Jailbreak direction Δμ", "direction"),
    "refusal": ("Refusal direction", "direction"),
    "alpaca_mean": ("Alpaca mean direction", "direction"),
    "wiki_mean": ("Wikipedia mean direction", "direction"),
    "alpaca_pca": ("Alpaca subspace (PCA-K)", "subspace"),
    "wiki_pca": ("Wikipedia subspace (PCA-K)", "subspace"),
    "hb_pca": ("HarmBench subspace (PCA-K)", "subspace"),
    "alpaca_acts": ("Alpaca activations as dictionary (mean of draws)", "subspace"),
    "wiki_acts": ("Wikipedia activations as dictionary (mean of draws)", "subspace"),
    "nr_acts": ("No Robots activations as dictionary (mean of draws)", "subspace"),
    "shuffled": ("Shuffled-label contrasts (mean of draws)", "subspace"),
    "concat": ("Concat [PersonaJB, raw] LR", "persona"),
    "JBShield": ("JBShield-style", "baseline"),
    "full_probe": ("Full-activation probe", "baseline"),
}
def det_label(d):
    if d in DETS: return DETS[d][0]
    m = re.match(r"(PCA|top|rand)_(\d+)$", d)
    if m: return {"PCA": f"Trait-matrix PCs, k={m[2]}", "top": f"Top-{m[2]} traits (by train AUC)", "rand": f"{m[2]} random traits (mean of 10 draws)"}[m[1]]
    m = re.match(r"tpc_(pca|svd|pcamean)_(\d+)$", d)
    if m: return {"pca": f"Trait PCs (centered), k={m[2]}", "svd": f"Trait SVD (uncentered), k={m[2]}", "pcamean": f"Mean trait + PCs, k={m[2]}"}[m[1]]
    m = re.match(r"lda_(.+)_(\d+)$", d)
    if m: return f"Persona-LDA α={m[1]}, k={m[2]}"
    m = re.match(r"pcak_(\d+)$", d)
    if m: return "Raw probe (no PCA)" if m[1] == "0" else f"Activation PCA({m[1]}) → LR"
    m = re.match(r"pm_(.+)$", d)
    if m: return {"LR_PersonaJB": "PersonaJB (LR)", "LR_Gaussian": "Random Gaussian proj. (LR, mean of 3)", "LR_raw": "Raw linear probe (LR)",
                  "PCA_LR": "PCA → LR (k, C tuned)", "DiffMeans_raw": "Difference of means (raw)", "DiffMeans_std": "Difference of means (standardized)",
                  "Concat_LR": "Concat [PersonaJB, raw] LR", "HGB_PersonaJB": "Gradient boosting on PersonaJB", "HGB_PCA128": "Gradient boosting on PCA-128",
                  "MLP_PersonaJB": "MLP on PersonaJB", "MLP_raw": "MLP on raw"}.get(m[1], m[1])
    return d

CONDS = {
    "full": "Full",
    "no_persona": "Persona component removed",
    "only_persona": "Only persona component",
    "no_random": "Random 240-dim span removed (null)",
    "no_random_span": "Random Gaussian span removed (null)",
    "no_nr": "No Robots span removed (NOT a valid control)",
    "no_wiki": "Wikipedia span removed",
    "no_wiki_P": "Wikipedia − persona span removed",
    "no_nr_P": "No Robots − persona span removed",
    "no_alpaca_P": "Alpaca − persona span removed",
    "no_hbpca": "HarmBench PCA span removed",
    "no_alpacapca": "Alpaca PCA span removed",
}

def fold(a):
    if a is None or (isinstance(a, float) and math.isnan(a)): return None
    return max(a, 1 - a)
def num(x):
    if x is None: return None
    try: x = float(x)
    except Exception: return None
    return None if math.isnan(x) or math.isinf(x) else round(x, 5)
def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None
def ci2(c): return [num(c[0]), num(c[1])] if c and len(c) == 2 and c[0] is not None else None
def rel(p):
    p = str(p)
    if p.startswith(str(PVJ) + "/") and PVJ == HERE.parent: return p[len(str(PVJ)) + 1:]
    return p.replace(str(HOME), "~")
def load(p):
    with open(p) as f: return json.load(f)

SOURCES = []   # {path, tab, section, setup, note}
def src(path, tab, section, setup=None, note=""):
    SOURCES.append({"path": rel(path), "tab": tab, "section": section, "setup": setup, "note": note})

SETUPS, RECS, HEAT = [], [], []
def setup(id, label, protocol, cols, **kw):
    s = dict(id=id, label=label, protocol=protocol, cols=cols, **kw); SETUPS.append(s); return s
def rec(s, m, d, auc=None, avg=None, ci=None, cond="full", train=None, readout=None, Cf=None, val=None, sel=None, dl=None, note=None, draws=None):
    r = {"s": s, "m": m, "d": d, "c": cond}
    if train is not None: r["t"] = train
    if readout is not None: r["r"] = readout
    if Cf is not None: r["C"] = Cf
    if auc: r["auc"] = {k: num(fold(v)) for k, v in auc.items() if v is not None}
    if avg is not None: r["avg"] = num(fold(avg))
    if ci: r["ci"] = ci2(ci)
    if val is not None: r["val"] = num(fold(val))
    if sel: r["sel"] = sel
    if dl: r["dl"] = dl
    if note: r["note"] = note
    if draws: r["draws"] = [num(x) for x in draws]
    RECS.append(r); return r

# ================================================================ DETECTION SOURCES
def q1(smoke=None):
    """Primary new source. Real outputs only (q1_{model}.json in q1_detection/out); a smoke file only with --smoke-q1."""
    outdir = FTO / "q1_detection/out"
    files = sorted(p for p in outdir.glob("q1_*.json") if not p.stem.endswith("_smoke")) if outdir.exists() else []
    is_smoke = False
    if smoke:
        files, is_smoke = [Path(smoke)], True
    S = setup("q1", "Q1 sweep: component removal (HB+WJB train, 5-fold CV readout selection)",
              "Train HB human-jailbreak + WJB pooled; readout (layer × tail position × C) chosen per detector by mean "
              "out-of-fold validation folded AUC over 5 folds grouped by HB template (WJB rows by behaviour), refit on all "
              "training rows; transfer = PAP/GCG/PAIR/GPTFuzz/PEZ; CI = HarmBench-behaviour cluster bootstrap (B=1000) of "
              "the average; Δ CI = paired bootstrap. Conditions remove the persona span (240 trait vectors) / a random "
              "240-dim span, or keep only the persona span, from every activation the detector sees. Readout facet: "
              "'re-selected' = layer/position/C chosen again under the condition; 'fixed' = the full detector's "
              "layer/position with C re-tuned.",
              FAM5, facets=["readout"], readouts={"reselect": "re-selected readout", "fixed": "fixed at full detector's (layer, position)"},
              pending=not files, expected=rel(outdir / "q1_gemma.json"), smoke=is_smoke, heat=True)
    src(FTO / "q1_detection/q1_sweep.py", "Detection", "Q1 sweep (schema)", "q1", "writer of q1_{model}.json; docstring = protocol")
    if not files:
        src(outdir / "q1_{model}.json", "Detection", "Q1 sweep", "q1", "PENDING: not on disk at build time; tab shows 'pending'")
        return
    for f in files:
        d = load(f); m = d["model"].replace("_smoke", "")
        src(f, "Detection + Persona share", "Q1 sweep", "q1", "SMOKE (dev only)" if is_smoke else "")
        S.setdefault("models_n", {})[m] = {"n_train": d.get("n_train"), "n_pos_train": d.get("n_pos_train"), "n_test": d.get("n_test"), "grid": d.get("grid")}
        TAIL = d["tail_tokens"]; sel = d["selected"]; dl = d["delta"]
        def selrec(det, cond, mode, entry, dkey):
            dd = None
            if cond != "full" and dkey in dl:
                x = dl[dkey]; dd = {"full": {"v": num(x["delta"]), "ci": ci2(x["ci95"])}}
            return rec("q1", m, det, auc=entry.get("test_auc"), avg=entry.get("avg"), ci=entry.get("avg_ci95"), cond=cond, readout=mode,
                       val=entry.get("val_auc"), sel={"L": entry.get("layer"), "p": entry.get("position"), "tok": entry.get("token"), "C": entry.get("C")} if "layer" in entry else None, dl=dd)
        dets = sorted({k.split("__")[0] for k in sel})
        base = [x for x in dets if not x.startswith("gauss_d")]
        for det in base:
            for cond in ("full", "no_persona", "no_random", "only_persona"):
                for mode in ("reselect", "fixed"):
                    k = f"{det}__{cond}__{'reselect' if cond == 'full' else mode}"
                    if k in sel: selrec(det, cond, mode, sel[k], f"{det}__{cond}__{mode}")
        # Gaussian: mean of the 3 draws per set (re-aggregation); avg + CI from the saved summary / delta entries
        draws = sorted(x for x in dets if x.startswith("gauss_d"))
        if draws:
            for cond in ("full", "no_persona", "no_random", "only_persona"):
                for mode in ("reselect", "fixed"):
                    ks = [f"{g}__{cond}__{'reselect' if cond == 'full' else mode}" for g in draws]
                    es = [sel[k] for k in ks if k in sel]
                    if not es: continue
                    auc = {s: mean([e["test_auc"].get(s) for e in es]) for s in d["test"]}
                    if cond == "full":
                        sm = sel.get("gauss__full__summary", {}); avg, ci, dd = sm.get("avg"), sm.get("avg_ci95"), None
                    else:
                        x = dl.get(f"gauss__{cond}__{mode}")
                        avg = x["avg"] if x else mean([e["avg"] for e in es]); ci = None
                        dd = {"full": {"v": num(x["delta"]), "ci": ci2(x["ci95"])}} if x else None
                    rec("q1", m, "gauss", auc=auc, avg=avg, ci=ci, cond=cond, readout=mode, val=mean([e["val_auc"] for e in es]),
                        sel={"per_draw": [f"L{e['layer']} p{e['position']} C={e['C']}" for e in es]}, dl=dd, draws=[e["avg"] for e in es])
        # heatmaps from cells
        cells = d["cells"]; L, P = d["layers"], list(range(len(TAIL)))
        groups = {}
        for k, v in cells.items():
            det, cond, Ls, ps = k.split("__"); det = "gauss" if det.startswith("gauss_d") else det
            groups.setdefault((det, cond), {}).setdefault((int(Ls[1:]), int(ps[1:])), []).append(v)
        for (det, cond), g in groups.items():
            val = [[num(mean([c["val_auc"] for c in g[(l, p)]])) if (l, p) in g else None for p in P] for l in L]
            avg = [[num(mean([mean([fold(c["test_auc"][s]) for s in d["test"]]) for c in g[(l, p)]])) if (l, p) in g else None for p in P] for l in L]
            HEAT.append({"s": "q1", "m": m, "d": det, "c": cond, "layers": L, "toks": TAIL, "metrics": {"val": val, "avg": avg}})
        # persona-share info -> share tab
        SHARE_Q1.append({"model": m, "info": d["info"], "layers": L, "toks": TAIL, "smoke": is_smoke, "file": rel(f)})

SHARE_Q1 = []

def hb_select():
    for cv in (False, True):
        suf = "_cv" if cv else ""
        sid = "hbsel" + suf
        S = setup(sid, "Readout selection " + ("(5-fold template-grouped CV)" if cv else "(single 80/20 val split)") + ": PersonaJB vs raw",
                  "hb_select.py" + (" --cv" if cv else "") + ". Train: HB human-jailbreak templates (Gemma: HB + WJB pooled). Both "
                  "detectors search every saved layer × tail position × C (1e-4..100); StandardScaler + LR. "
                  + ("Validation = 5-fold CV grouped by HB template (WJB rows by behaviour), mean fold AUC. " if cv else
                     "Validation = one behaviour-grouped 80/20 split (HB alone: also template-disjoint). ")
                  + "Selected cell refit on all training data. Transfer = every other set (Llama/OLMo/Qwen include WJB). "
                  "CI: HarmBench-behaviour cluster bootstrap B=1000; Δ = PersonaJB − raw (paired).",
                  ["wjb"] + FAM5, heat=True)
        for m in ("llama", "gemma", "olmo", "qwen"):
            f = DET / f"hb_select_out/hb_select_{m}{suf}.json"
            if not f.exists(): continue
            d = load(f); src(f, "Detection", S["label"], sid)
            dd = d["delta"]
            for det, e in d["selected"].items():
                x = {"raw": {"v": num(dd["delta"]), "ci": ci2(dd["ci95"])}} if det == "PersonaJB" else None
                rec(sid, m, det, auc=e["test_auc"], avg=e["avg"], ci=e["avg_ci95"], val=e["val_auc"],
                    sel={"L": e["layer"], "p": e["position"], "tok": e["token"], "C": e["C"]}, dl=x, train="hbwjb" if d["train"] == ["hb", "wjb"] else "hb")
            L, TAIL = d["layers"], d["tail_tokens"]
            for det in ("PersonaJB", "raw"):
                val = [[num(d["grid"].get(f"{det}__L{l}__p{p}", {}).get("val_auc")) for p in range(len(TAIL))] for l in L]
                avg = [[num(mean([fold(a) for a in d["grid"][f"{det}__L{l}__p{p}"]["test_auc"].values()])) if f"{det}__L{l}__p{p}" in d["grid"] else None
                        for p in range(len(TAIL))] for l in L]
                HEAT.append({"s": sid, "m": m, "d": det, "c": "full", "layers": L, "toks": TAIL, "metrics": {"val": val, "avg": avg}})
        S["trains"] = {"hb": "HB human jailbreaks", "hbwjb": "HB + WJB pooled (Gemma)"}

def single_set():
    """persona_four.json (PersonaJB / raw / Gaussian, PersonaJB_filtered for OLMo) + persona_variants.json (variants).
    Same protocol and identical raw / PersonaJB numbers (checked at build time)."""
    P, V = load(DET / "persona_four.json"), load(DET / "persona_variants.json")
    F = load(DET / "four_models.json")
    S = setup("single", "Single-set training (paper readout, C on train-set val split)",
              "Train on ONE set (HB human jailbreaks, WJB, or one attack family); StandardScaler + LR, C from 1e-4..100 chosen "
              "on a behaviour-grouped validation split of the training set (HB also template-disjoint), refit on the whole "
              "training set. Readout: Llama L16, Gemma L30, Qwen L48, OLMo L16; last prompt token. Transfer per training "
              "set = mean folded AUC over the attack families GCG/PAIR/PAP/GPTFuzz/PEZ other than itself. CI: HarmBench-"
              "behaviour cluster bootstrap B=1000; Δ CIs Bonferroni-corrected as saved. Gaussian = K unit random "
              "directions, K = trait count, mean of 3 draws.",
              ["hb", "wjb"] + FAM5, facets=["train"], trains={t: SETS[t] if t != "hb" else "HB human jailbreaks" for t in ["hb", "wjb"] + FAM5} | {"avg": "mean over the 7 training sets"})
    src(DET / "persona_four.json", "Detection", S["label"], "single")
    src(DET / "persona_variants.json", "Detection", S["label"], "single", "PCA_k / top_k / rand_k / concat variants")
    src(DET / "four_models.json", "Detection", S["label"], "single", "raw/Gaussian: identical to persona_four (verified at build); only C values read")
    mism = 0
    for m in MODELS:
        for t in ["hb", "wjb"] + FAM5 + ["avg"]:
            c = P["cells"].get(f"{m}__{t}")
            if not c: continue
            for det, dk in (("PersonaJB", "PersonaJB"), ("raw", "raw"), ("gauss", "Gaussian"), ("PersonaJB_filtered", "PersonaJB_filtered")):
                if dk not in c: continue
                if det == "gauss":
                    pf = [P["per_family"].get(f"{m}__Gaussian{i}__{t}") for i in range(3)]
                    auc = {s: mean([x.get(s) for x in pf if x]) for s in FAM5} if t != "avg" and all(pf) else None
                else:
                    auc = P["per_family"].get(f"{m}__{dk}__{t}") if t != "avg" else None
                dl = {}
                if det in ("PersonaJB", "PersonaJB_filtered"):
                    if f"{dk}-raw" in c: dl["raw"] = {"v": num(c[f"{dk}-raw"]), "ci": ci2(c[f"{dk}-raw_ci_bonf"]), "kind": "Bonferroni"}
                    if f"{dk}-Gaussian" in c: dl["gauss"] = {"v": num(c[f"{dk}-Gaussian"]), "ci": ci2(c[f"{dk}-Gaussian_ci_bonf"]), "kind": "Bonferroni"}
                Cs = None
                if t != "avg":
                    if det.startswith("PersonaJB"): Cs = (P["C"].get(m) or {}).get(f"{dk}__{t}") or (V["C"].get(m) or {}).get(f"{dk}__{t}")
                    elif det == "raw": Cs = (F["C"].get(m) or {}).get(f"raw__{t}")
                    elif det == "gauss": Cs = "/".join(str((F["C"].get(m) or {}).get(f"Gaussian{i}__{t}")) for i in range(3))
                rec("single", m, det, auc=auc, avg=c[dk], ci=c.get(f"{dk}_ci95"), train=t, dl=dl or None, sel={"C": Cs} if Cs is not None else None)
        # variants
        for key, c in V["cells"].items():
            mm, det, t = key.split("__")
            if mm != m or det in ("raw", "PersonaJB"):
                if mm == m and det in ("raw", "PersonaJB") and abs(c["auc"] - P["cells"][f"{m}__{t}"][det]) > 1e-9: mism += 1
                continue
            auc = V["per_family"].get(f"{m}__{det}__{t}") if t != "avg" else None
            Cs = (V["C"].get(m) or {}).get(f"{det}__{t}") if t != "avg" else None
            rec("single", m, det, auc=auc, avg=c["auc"], ci=c.get("ci95"), train=t,
                dl={"raw": {"v": num(c["delta_vs_raw"]), "ci": ci2(c["delta_ci_bonf"]), "kind": "Bonferroni"}},
                sel={"C": Cs} if Cs is not None else None)
    if mism: print(f"WARNING single_set: {mism} raw/PersonaJB cells differ between persona_variants and persona_four", file=sys.stderr)
    S["top_traits"] = V.get("top_traits")
    S["default_dets"] = ["PersonaJB", "PersonaJB_filtered", "raw", "gauss", "concat", "PCA_229", "PCA_240", "top_10", "rand_10"]

def dom_compare():
    d = load(DET / "dom_compare/dom_olmo_compare.json")
    S = setup("dom", "OLMo-3-7B on Dominic's activations (single-set training)",
              "Same protocol as the single-set setup (C tuned on a behaviour-grouped val split of the training set, transfer "
              "= mean folded AUC over the other families) but on Dominic's OLMo-3-7B activations (RCP *_activations_olmo3, "
              "L16 last token). No GCG (his GCG run never finished). Δ CIs: paired behaviour-cluster bootstrap B=1000 (95%, "
              "not Bonferroni).", ["hb", "wjb", "pair", "pap", "gptfuzz", "pez"], facets=["train"],
              trains={t: SETS[t] for t in ["hb", "wjb", "pair", "pap", "gptfuzz", "pez"]} | {"avg": "mean over training sets"})
    src(DET / "dom_compare/dom_olmo_compare.json", "Detection", S["label"], "dom")
    for t, c in d["cells"].items():
        for det, dk in (("PersonaJB", "PersonaJB"), ("raw", "raw"), ("gauss", "Gaussian")):
            if det == "gauss":
                pf = [d["per_family"].get(f"Gaussian{i}__{t}") for i in range(3)]
                auc = {s: mean([x.get(s) for x in pf]) for s in pf[0]} if t != "avg" and all(pf) else None
            else: auc = d["per_family"].get(f"{dk}__{t}") if t != "avg" else None
            dl = None
            if det == "PersonaJB":
                dl = {"raw": {"v": num(c["PersonaJB-raw"]), "ci": ci2(c.get("PersonaJB-raw_ci95"))}, "gauss": {"v": num(c["PersonaJB-Gaussian"]), "ci": ci2(c.get("PersonaJB-Gaussian_ci95"))}}
            Cs = d["C"].get(f"{dk}__train_{t}") if det != "gauss" else "/".join(str(d["C"].get(f"Gaussian{i}__train_{t}")) for i in range(3))
            rec("dom", "olmo", det, auc=auc, avg=c[dk], train=t, dl=dl, sel={"C": Cs} if t != "avg" else None)

def olmo_layer_sweep():
    d = load(DET / "olmo_probe.json")
    S = setup("olmo_layers", "OLMo-3-7B raw probe: layer sweep (single-set training)",
              "olmo_probe.py: raw-activation LR, C chosen on a behaviour-grouped val split of the training set, last prompt "
              "token, every saved layer; transfer = mean folded AUC over the other families (GCG/PAIR/PAP/GPTFuzz/PEZ). No "
              "bootstrap for the layer sweep (CIs only at the main layer 16, which equals the single-set setup).",
              [], facets=["train"], trains={t: SETS[t] for t in ["hb", "wjb"] + FAM5} | {"avg": "mean over training sets"}, readout_axis=True)
    src(DET / "olmo_probe.json", "Detection", S["label"], "olmo_layers", "layer_sweep only; 'transfer' at L16 duplicates persona_four raw")
    for L, row in d["layer_sweep"].items():
        for t, v in row.items(): rec("olmo_layers", "olmo", "raw", avg=v, train=t, readout=f"L{L}")
    S["readouts"] = {f"L{L}": f"layer {L}" for L in d["layer_sweep"]}

def paper50():
    """trait_pc_sweep summaries: the published PersonaJB-row protocol, 50 HB pool splits."""
    S = setup("paper50", "Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries)",
              "trait_pc_sweep.py = the published PersonaJB row (run_all_traits_sweep_v2 logreg_l2_C10.0_raw): per seed 0..49 a "
              "behaviour × template pool split (70/30) of HarmBench human jailbreaks; StandardScaler + LR(C fixed per file, "
              "C=10 = paper) refit on the train pool. Gemma folds all WildJailbreak rows into the fit. Readout Llama L16 / "
              "Gemma L30, last prompt token. Cell = mean over seeds of folded AUC; avg transfer = mean over GCG/PAIR/PAP/"
              "GPTFuzz/PEZ; CI = 1.96·sd/√50 of per-seed averages. 'val' = inner 15% val split AUC. 'val-C' = C picked per "
              "seed on that val split (aggregate.json). Draw-based dictionaries (Alpaca/Wikipedia/No Robots activations, "
              "shuffled contrasts, Gaussian) are shown as the mean of 5 draws (per-draw avgs in the tooltip, no CI).",
              ["hb_test"] + FAM5, facets=["C"])
    Cs = {}
    TPC = FTO / "trait_pc_sweep"
    seen = set()
    groups = {}   # (m, Cf, det) -> list of (cfgname, cfg)
    for f in sorted(TPC.glob("summary_*.json")):
        d = load(f); m = d["model"]; C = d.get("C") or 10.0
        Cf = f"C={C:g}"; Cs[Cf] = f"fixed C={C:g}" + (" (paper)" if C == 10 else "")
        src(f, "Detection", S["label"], "paper50", f"C={C:g}; signed fields dropped")
        for n, c in d["configs"].items():
            if (m, Cf, n) in seen: continue
            seen.add((m, Cf, n))
            b, k = c["basis"], c["k"]
            mm = re.match(r"(acts|noise|gauss|wiki|nr)_d(\d)$", b)
            if b == "traits": det = "PersonaJB"
            elif b in ("pca", "svd", "pcamean"): det = f"tpc_{b}_{k}"
            elif b.startswith("lda_"): det = f"lda_{b[4:].lstrip('a')}_{k}"
            elif b == "alpaca_pca": det = "alpaca_pca"
            elif b == "hb_pca": det = "hb_pca"
            elif mm: det = {"acts": "alpaca_acts", "noise": "shuffled", "gauss": "gauss", "wiki": "wiki_acts", "nr": "nr_acts"}[mm[1]]
            else: det = b
            groups.setdefault((m, Cf, det), []).append(c)
    for (m, Cf, det), cs in groups.items():
        fam = {"hb_test": "human_test", "gcg": "GCG", "pair": "PAIR", "pap": "PAP", "gptfuzz": "GPTFuzz", "pez": "PEZ"}
        if len(cs) == 1:
            c = cs[0]; ci = c["avg_transfer_ci95"]
            rec("paper50", m, det, auc={s: c["mean"][k] for s, k in fam.items()}, avg=c["avg_transfer"], ci=[c["avg_transfer"] - ci, c["avg_transfer"] + ci],
                val=c["mean"].get("val"), Cf=Cf)
        else:
            rec("paper50", m, det, auc={s: mean([c["mean"][k] for c in cs]) for s, k in fam.items()}, avg=mean([c["avg_transfer"] for c in cs]),
                val=mean([c["mean"].get("val") for c in cs]), Cf=Cf, draws=[c["avg_transfer"] for c in cs], note=f"mean of {len(cs)} draws")
    A = load(TPC / "aggregate.json"); src(TPC / "aggregate.json", "Detection", S["label"], "paper50", "val-C rows (C chosen per seed on inner val)")
    Cs["val-C"] = "C chosen per seed on the inner val split"
    for m, O in A.items():
        for b in ("pca", "svd", "traits"):
            for r in O[b]:
                v = r["valC"]; det = "PersonaJB" if b == "traits" else f"tpc_{b}_{r['k']}"
                rec("paper50", m, det, auc={"hb_test": v["hb"], **{s.lower(): v["fam"][s] for s in v["fam"]}}, avg=v["xfer"], ci=[v["xfer"] - v["ci"], v["xfer"] + v["ci"]],
                    Cf="val-C", sel={"C": f"modal {v['C_mode']}"})
        for b in ("pca", "svd"):
            j = O.get(b + "_valkC")
            if j: rec("paper50", m, f"tpc_{b}_valk", auc={"hb_test": j["hb"]}, avg=j["xfer"], ci=[j["xfer"] - j["ci"], j["xfer"] + j["ci"]], Cf="val-C",
                      sel={"k": f"median {j['k_median']:.0f}, range {j['k_range']}"})
    DETS["tpc_pca_valk"] = ("Trait PCs (centered), k and C chosen on val", "persona")
    DETS["tpc_svd_valk"] = ("Trait SVD, k and C chosen on val", "persona")
    S["Cs"] = Cs; S["default_C"] = "C=10"
    S["default_dets"] = ["PersonaJB", "tpc_pca_valk", "alpaca_pca", "hb_pca", "alpaca_acts", "wiki_acts", "nr_acts", "shuffled", "gauss", "tpc_pca_228", "tpc_pca_239", "tpc_svd_229", "tpc_svd_240"]

def removal50():
    S = setup("removal50", "Persona-removal controls (50 HB pool splits, C=10, dictionary detectors)",
              "remove_persona_all_controls.py + remove_nonpersona_span.py: the PersonaJB recipe (StandardScaler + LR C=10, the "
              "same 50 pool splits; Gemma: + all WJB rows in the fit) run on 'dictionaries' of K vectors (K=229 Llama / 240 "
              "Gemma) instead of the trait vectors. Condition = a span projected out of every dictionary vector (then "
              "renormalized). Folded AUC; HB test + mean of the 5 transfer families (per-family AUCs not saved). Draw-based "
              "dictionaries = mean of 3 draws (per-draw values in the tooltip; no CI saved). Readout Llama L16 / Gemma L30 last "
              "token. Not the same code path as the trait-PC sweep: numbers differ from it by up to ~0.003, so they are kept in "
              "a separate setup.", ["hb_test"])
    MAP = {"persona": "no_persona", "No Robots span": "no_nr", "random span": "no_random_span", "Wikipedia": "no_wiki",
           "Wikipedia - P": "no_wiki_P", "No Robots - P": "no_nr_P", "Alpaca - P": "no_alpaca_P", "random": "no_random_span"}
    DN = {"Wikipedia acts": "wiki_acts", "No Robots acts": "nr_acts", "Alpaca acts": "alpaca_acts", "Shuffled contrasts": "shuffled",
          "Random Gaussian": "gauss", "Alpaca PCA-229": "alpaca_pca", "HarmBench PCA-229": "hb_pca", "Alpaca PCA-240": "alpaca_pca", "HarmBench PCA-240": "hb_pca"}
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        A = load(base / "remove_persona_all_controls.json"); B = load(base / "remove_nonpersona_span.json")
        src(base / "remove_persona_all_controls.json", "Detection", S["label"], "removal50", "conditions: persona / No Robots span / random span")
        src(base / "remove_nonpersona_span.json", "Detection", S["label"], "removal50", "conditions: Wikipedia, Wikipedia−P, No Robots−P, Alpaca−P spans ('persona' and 'random' duplicate the file above, checked)")
        merged = {}
        for n, outs in A.items(): merged[n] = [dict(o) for o in outs]
        for n, outs in B["dicts"].items():
            for i, o in enumerate(outs):
                for k, v in o.items():
                    if k in ("whole", "persona", "random"):
                        ref = merged[n][i]["whole" if k == "whole" else ("persona" if k == "persona" else "random span")]
                        if [round(x, 9) for x in ref] != [round(x, 9) for x in v]: print(f"WARNING removal50 {m} {n} {k} differs between files", file=sys.stderr)
                        continue
                    merged[n][i][k] = v
        for n, outs in merged.items():
            det = DN[n]
            w = [o["whole"] for o in outs]
            rec("removal50", m, det, auc={"hb_test": mean([x[0] for x in w])}, avg=mean([x[1] for x in w]), draws=[x[1] for x in w] if len(w) > 1 else None)
            for k, cond in MAP.items():
                vs = [o[k] for o in outs if k in o]
                if not vs: continue
                rec("removal50", m, det, cond=cond, auc={"hb_test": mean([x[1] for x in vs])}, avg=mean([x[2] for x in vs]),
                    draws=[x[2] for x in vs] if len(vs) > 1 else None, sel={"share_removed": num(mean([x[0] for x in vs]))})
        for sn, st in B["spans"].items():
            SHARE_SPANS.append({"model": m, "span": sn, **{k: num(v) for k, v in st.items()}, "file": rel(base / "remove_nonpersona_span.json")})
    S["conds"] = ["full", "no_persona", "no_random_span", "no_wiki", "no_wiki_P", "no_nr_P", "no_alpaca_P", "no_nr"]
SHARE_SPANS = []

def removal10():
    S = setup("removal10", "Persona-part tests (10 HB pool splits, C=10, Llama)",
              "pcs_persona_component.py (Alpaca / HarmBench PCA-229 dictionaries) and persona_part_test.py (Wikipedia / No Robots "
              "/ Alpaca activation dictionaries, draw 0): PersonaJB recipe (StandardScaler + LR C=10) on 10 pool splits only "
              "(not the 50 of the paper protocol), Llama L16 last token, HB train only. Condition = the named span's component "
              "removed from every dictionary vector. HB test + mean of 5 transfer families, folded. persona_part_test does not "
              "save the unremoved ('full') detector, so those cells are missing.", ["hb_test"])
    d = load(NSC / "pcs_persona_component.json"); src(NSC / "pcs_persona_component.json", "Detection", S["label"], "removal10")
    for pn, det in (("alpaca_pca", "alpaca_pca"), ("hb_pca", "hb_pca")):
        e = d[pn]; rec("removal10", "llama", det, auc={"hb_test": e["whole"][0]}, avg=e["whole"][1])
        for k, cond in (("persona traits", "no_persona"), ("Wikipedia acts", "no_wiki"), ("No Robots acts", "no_nr"), ("random Gaussian", "no_random_span")):
            vs = e[k]; rec("removal10", "llama", det, cond=cond, auc={"hb_test": mean([x[1] for x in vs])}, avg=mean([x[2] for x in vs]),
                           draws=[x[2] for x in vs] if len(vs) > 1 else None, sel={"share_removed": num(mean([x[0] for x in vs]))})
    p = load(NSC / "persona_part_test.json"); src(NSC / "persona_part_test.json", "Detection", S["label"], "removal10", "det_out_hb / det_out_xf only; single-vector AUCs go to Persona share")
    DN = {"Wikipedia": "wiki_acts_d0", "No Robots": "nr_acts_d0", "Alpaca": "alpaca_acts_d0"}
    CN = {"Persona traits": "no_persona", "Random Gaussian": "no_random_span", "HarmBench PCA-229": "no_hbpca", "Alpaca PCA-229": "no_alpacapca"}
    for k, v in p.items():
        dn, sn = k.split("|")
        rec("removal10", "llama", DN[dn], cond=CN[sn], auc={"hb_test": v["det_out_hb"]}, avg=v["det_out_xf"], sel={"share_removed": num(v["share"])})
    DETS.update({"wiki_acts_d0": ("Wikipedia activations (draw 0)", "subspace"), "nr_acts_d0": ("No Robots activations (draw 0)", "subspace"),
                 "alpaca_acts_d0": ("Alpaca activations (draw 0)", "subspace")})
    S["conds"] = ["full", "no_persona", "no_random_span", "no_wiki", "no_hbpca", "no_alpacapca", "no_nr"]

def release():
    S = setup("release", "Clean-repo release run (50 splits, config chosen on train-pool val)",
              "~/persona-jailbreaks scripts/4_detect.py on the release data (~/pjb_release_staging): 50 behaviour × template "
              "splits of HarmBench (70/30); a stratified 15% of each training pool selects one configuration per method (C × "
              "class weight grid; highest val AUC averaged over splits), refit on the training pool; Gemma adds WJB to every "
              "fitting set. Folded AUC per split. Transfer CI: prompt resampling within each family (1000 draws). Llama PAIR "
              "here excludes the 400 unattacked direct requests that the paper's PAIR set included, so Llama PAIR / avg "
              "differ from the paper. Only activation methods were run.", ["hb_test"] + FAM5)
    N = {"PersonaJB": "PersonaJB", "Activation probe": "raw", "PCA": "PCA", "Jailbreak direction": "jb_dir", "Refusal direction": "refusal", "JBShield-style": "JBShield"}
    for m in ("llama", "gemma"):
        f = RELROOT / f"derived/{m}/detection.json"; d = load(f); src(f, "Detection", S["label"], "release")
        for k, e in d.items():
            rec("release", m, N.get(k, k), auc={"hb_test": e["harmbench"], **e["families"]}, avg=e["transfer"], ci=e.get("transfer_ci"), val=e.get("val_auc"),
                sel={"C": e["config"].get("C"), "cw": e["config"].get("cw")} if e.get("config") else None)

def qwen_sources():
    Q = FTO / "qwen3_32b"
    d = load(Q / "transfer/detect_transfer.json")
    S = setup("qwen_pos", "Qwen3-32B L48 per tail position (49 pool splits, C=10, HB+WJB)",
              "qwen3_32b/transfer/detect_transfer.py (protocol of stage2/qwen_detect.py): StandardScaler + LR C=10 on 240 "
              "projections; HB human-jailbreak train pool + all WJB; 50 pool splits with both classes (49 used); folded AUC on HB "
              "test and each of Qwen's own GCG/PAIR/PEZ/GPTFuzz test cases + PAP. Full probe: LR C=0.01 on 5120 dims, 10 splits. "
              "Gaussian = mean of 3 draws. No CI saved.", ["hb_test"] + FAM5, facets=["readout"])
    src(Q / "transfer/detect_transfer.json", "Detection", S["label"], "qwen_pos")
    S["readouts"] = {}
    for pos, e in d["positions"].items():
        r = f"p{pos} {e['token']!r}"; S["readouts"][r] = f"position {pos} ({e['token']!r})"
        for dk, det in (("PersonaJB", "PersonaJB"), ("Gaussian", "gauss"), ("full probe", "full_probe")):
            x = e.get(dk)
            if not x: continue
            rec("qwen_pos", "qwen", det, auc={"hb_test": x["hb"], **{s: x[s] for s in FAM5 if s in x}}, avg=x.get("transfer_mean"), readout=r)
    # layer x position sweep -> heatmaps only
    sw = load(Q / "sweep/out/sweep.json"); src(Q / "sweep/out/sweep.json", "Detection", "Layer × position heatmap (Qwen sweep)", "qwen_sweep")
    setup("qwen_sweep", "Qwen3-32B layer × position sweep (20 pool splits, C=10)",
          "sweep/detect_sweep.py: StandardScaler + LR C=10 on 240 projections (PersonaJB / Gaussian draws), full probe LR C=0.01 "
          "(5 splits). Training pool HB human-jailbreak only or HB + all WJB (condition facet). Metrics: HB template/behaviour-"
          "disjoint test (20 splits), HB random 5-fold CV, mean transfer over Qwen's own GCG/PAIR/PAP/GPTFuzz/PEZ. Heatmap only.",
          [], heat=True, heat_only=True)
    L, TAIL = sw["layers"], sw["tail_tokens"]
    for det, dk in (("PersonaJB", "PersonaJB"), ("gauss", "Gaussian"), ("full_probe", "fullprobe")):
        for tr in ("hb", "hbwjb"):
            mets = {}
            for met, key in (("hb_disjoint", "hb_disjoint"), ("hb_randomCV", "hb_randomCV"), ("avg", "transfer_mean")):
                mets[met] = [[num(sw["cells"].get(f"L{l}_p{p}", {}).get(f"{dk}__{tr}", {}).get(key)) for p in range(len(TAIL))] for l in L]
            HEAT.append({"s": "qwen_sweep", "m": "qwen", "d": det, "c": tr, "layers": L, "toks": TAIL, "metrics": mets})
    # probe_methods: single-set + LOFO method sweep
    pm = load(Q / "probe_methods/out/probe_methods.json")
    S = setup("probe_methods", "Probe-method sweep: single-set and leave-one-family-out (Llama L16, Qwen L48)",
              "probe_methods.py: every hyperparameter tuned on a behaviour-grouped validation split of the TRAINING data (HB alone: "
              "behaviour- and template-disjoint), refit on all training data. Regimes: single (train on one set, test on the "
              "attack families other than itself) and LOFO (train on every set but one family, sets weighted equally, test on "
              "the held-out family; columns = held-out family). CIs: behaviour-clustered bootstrap B=1000; Δ vs raw LR "
              "Bonferroni (m=44). Last prompt token.", ["hb", "wjb"] + FAM5, facets=["train"],
              trains={t: SETS[t] for t in ["hb", "wjb"] + FAM5} | {"avg": "single-set, mean over training sets", "lofo": "leave one family out"})
    src(Q / "probe_methods/out/probe_methods.json", "Detection", S["label"], "probe_methods")
    T, Cc = pm["table"], pm["cells"]
    for key, e in T.items():
        m, meth, reg = key.split("__")
        if reg.startswith("lofo_") and reg != "lofo_avg": continue
        det = "pm_" + meth
        dl = {"raw": {"v": num(e.get("delta_vs_LR_raw")), "ci": ci2(e.get("delta_ci_bonf")), "kind": "Bonferroni"}} if meth != "LR_raw" else None
        if reg == "lofo_avg":
            auc = {f: (T.get(f"{m}__{meth}__lofo_{f}") or {}).get("auc") for f in FAM5}
            rec("probe_methods", m, det, auc=auc, avg=e["auc"], ci=e.get("ci95"), train="lofo", dl=dl)
        elif reg == "single_avg":
            rec("probe_methods", m, det, avg=e["auc"], ci=e.get("ci95"), train="avg", dl=dl)
        else:
            t = reg[len("single_"):]
            if meth == "LR_Gaussian":
                cs = [Cc.get(f"{m}__single_{t}__LR_Gaussian{i}") for i in range(3)]
                auc = {s: mean([c["per_test"].get(s) for c in cs if c]) for s in FAM5} if all(cs) else None
                cfg = "/".join(str(c["cfg"]) for c in cs if c)
            else:
                c = Cc.get(f"{m}__single_{t}__{meth}"); auc = c["per_test"] if c else None; cfg = c["cfg"] if c else None
            rec("probe_methods", m, det, auc=auc, avg=e["auc"], ci=e.get("ci95"), train=t, dl=dl, sel={"C": cfg} if cfg else None)

def pca_k():
    D = FTO / "pca_k_sweep"
    for m, sid, lab, prot in (
        ("llama", "pcak_llama", "Activation PCA(k) sweep, Llama protocol (50 seeds)",
         "pca_k_sweep.py llama: per seed a behaviour × template pool split (70/30) → StandardScaler(train) → PCA(k) on the "
         "standardized train pool → LR(C=1). Shown: auc_sym = folded AUC on the FULL test set for every cell (the paper's "
         "mixed per-cell protocol with signed / undersampled cells is not shown). Mean over 50 seeds; avg CI = 1.96·sd/√50 of "
         "per-seed averages (re-aggregated from the saved per-seed AUCs). k=0 = raw probe. Llama L16 last token, HB train only."),
        ("gemma", "pcak_gemma", "Activation PCA(k) sweep, Gemma protocol (50 seeds)",
         "pca_k_sweep.py gemma (run_all_traits_sweep_v2 --feature_type pca): PCA(k) fit ONCE on all HB human-jailbreak "
         "activations (unstandardized); per seed pool split, train pool = HB train + all WJB; StandardScaler + LR(C=1, "
         "class_weight=balanced). Folded AUC (auc_sym) on the full set; mean over 50 seeds; CI = 1.96·sd/√50. Gemma L30 last token.")):
        S = setup(sid, lab, prot, ["hb_test"] + FAM5, facets=["C"], sweep_k=True)
        Cs = {}
        for f in sorted(D.glob(f"*{m}*.json")):
            d = load(f)
            C = d.get("C"); cw = d.get("class_weight")
            Cf = "C=1 (paper row)" if C is None else f"C={C:g}" + ("" if cw else ", unbalanced") if m == "gemma" else ("C=1 (paper row)" if C is None else f"C={C:g}")
            if m == "gemma" and C is not None: Cf = f"C={C:g}, " + ("balanced" if cw == "balanced" else "unbalanced")
            Cs[Cf] = Cf; src(f, "Detection", lab, sid, "auc_sym only (signed auc / auc_paper dropped)")
            for k, e in d["per_k"].items():
                pf = e["per_family"]; n = len(pf["GCG"]["auc_sym"])
                fam = {"hb_test": "human_test", "gcg": "GCG", "pair": "PAIR", "pap": "PAP", "gptfuzz": "GPTFuzz", "pez": "PEZ"}
                per_seed = [mean([fold(pf[F]["auc_sym"][i]) for F in ("GCG", "PAIR", "PAP", "GPTFuzz", "PEZ")]) for i in range(n)]
                mu = mean(per_seed); sd = (sum((x - mu) ** 2 for x in per_seed) / (n - 1)) ** .5; h = 1.96 * sd / n ** .5
                rec(sid, m, f"pcak_{k}", auc={s: mean([fold(x) for x in pf[F]["auc_sym"]]) for s, F in fam.items()}, avg=mu, ci=[mu - h, mu + h], Cf=Cf,
                    sel={"k": int(k), "expl_var": num(e.get("explained_var"))})
        S["Cs"] = Cs; S["default_C"] = "C=1 (paper row)"

def few_dirs():
    f = NSC / "few_directions.json"
    if not f.exists(): return
    FEW.append(load(f)); src(f, "Detection", "Few directions (k best train-AUC directions per dictionary)", None, "see few_directions parser")
FEW = []

def complement():
    d = load(DET / "complement/complement.json"); src(DET / "complement/complement.json", "Detection", "Complementarity: does PersonaJB catch what the raw probe misses?", None)
    COMPLEMENT.update(d)
COMPLEMENT = {}

def dir50():
    S = setup("dir50", "Single directions with persona component removed (50 HB pool splits)",
              "appendix_controls_data.py auc block: score = h·v for a single direction v (no fitted classifier), Llama L16 / Gemma "
              "L30 last token; Δμ re-estimated on each split's train rows. Folded AUC, HB test mean over 50 pool splits (CI = "
              "1.96·sd/√50) + mean of the 5 transfer families (per-family not saved). Conditions: v itself, its persona-span "
              "component only, persona component removed, a random K-dim Gaussian span removed (5 spans). Null directions: "
              "stratified shuffles (the clean null), pooled shuffles (inflated, kept for completeness), random Gaussian.", ["hb_test"])
    DN = {"refusal": "refusal", "dmu": "jb_dir", "shuf_refusal": "shuf_refusal", "shuf_dmu": "shuf_dmu", "strat_refusal": "strat_refusal",
          "strat_dmu": "strat_dmu", "shuf_alpaca": "shuf_alpaca", "random_direction": "random_dir"}
    DETS.update({"shuf_refusal": ("Shuffled refusal (pooled permutation; inflated null)", "null"), "shuf_dmu": ("Shuffled Δμ (pooled permutation; inflated null)", "null"),
                 "strat_refusal": ("Stratified-shuffle refusal (clean null)", "null"), "strat_dmu": ("Stratified-shuffle Δμ (clean null)", "null"),
                 "shuf_alpaca": ("Shuffled Alpaca contrast", "null"), "random_dir": ("Random Gaussian direction", "null")})
    CM = {"full": "full", "persona_component": "only_persona", "persona_removed": "no_persona", "random_removed": "no_random_span"}
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        f = base / "appendix_controls_data.json"; d = load(f); src(f, "Detection + Persona share", S["label"] + " / direction share", "dir50", "auc block → Detection; share blocks → Persona share")
        for k, conds in d["auc"].items():
            for ck, e in conds.items():
                h = e.get("hb_ci")
                rec("dir50", m, DN[k], cond=CM[ck], auc={"hb_test": e["hb"]}, avg=e["transfer"], sel={"hb_ci95": [num(e["hb"] - h), num(e["hb"] + h)]} if h is not None else None)
    S["conds"] = ["full", "no_persona", "only_persona", "no_random_span"]

def few_dirs():
    f = NSC / "few_directions.json"; d = load(f)
    S = setup("few", "Few directions: k best train-AUC directions per dictionary (Llama, 50 splits, C=10)",
              "few_directions.py: from each K=229 dictionary pick the k directions with the best train-split AUC, fit the PersonaJB "
              "recipe (StandardScaler + LR C=10) on them, 50 pool splits, Llama L16. Only the FOLDED transfer AUC (xf_folded, mean "
              "of 5 families) is shown; the file's HB and xf_signed fields are signed AUCs and are dropped. Draw-based "
              "dictionaries: mean of 5 draws.", [], facets=["readout"], readouts={}, readout_axis=True)
    src(f, "Detection", S["label"], "few", "xf_folded only; hb and xf_signed are signed AUC → dropped")
    groups = {}
    for n, ks in d.items():
        mm = re.match(r"(acts|noise|gauss)_d\d$", n)
        det = {"persona": "PersonaJB", "alpaca_pca": "alpaca_pca", "hb_pca": "hb_pca"}.get(n) or {"acts": "alpaca_acts", "noise": "shuffled", "gauss": "gauss"}[mm[1]]
        for k, e in ks.items(): groups.setdefault((det, k), []).append(mean([fold(x) for x in e["xf_folded"]]))
    for (det, k), vs in groups.items():
        S["readouts"][f"k={k}"] = f"k = {k} directions"
        rec("few", "llama", det, avg=mean(vs), readout=f"k={k}", draws=vs if len(vs) > 1 else None)

def single_vec():
    f = NSC / "single_vector_auc.json"; d = load(f)
    S = setup("single_vec", "Single ordinary vectors as detectors (h·v, in-sample, Llama)",
              "single_vector_auc.py: every vector v of a dictionary used alone, score h·v, no fitting; folded AUC on all HB rows "
              "(in-sample) and mean over the 5 transfer families. Shown: MEDIAN over the vectors (229 traits / 1145 per "
              "activation set / 1145 Gaussian), re-aggregated from the saved per-vector AUCs. Llama L16 last token.", ["hb_test"])
    src(f, "Detection", S["label"], "single_vec")
    def med(xs):
        xs = sorted(fold(x) for x in xs); n = len(xs); return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    DN = {"Persona traits": "sv_traits", "Wikipedia activations": "sv_wiki", "No Robots activations": "sv_nr", "Alpaca activations": "sv_alpaca", "Random Gaussian": "sv_gauss"}
    for k, e in d.items():
        DETS[DN[k]] = (f"{k} (median single vector)", "single")
        rec("single_vec", "llama", DN[k], auc={"hb_test": med(e["hb"])}, avg=med(e["transfer"]), note=f"median of {len(e['hb'])}")

def normality_det():
    f = NSC / "normality_in_persona.json"; d = load(f)
    S = setup("normality", "Benign-vs-harmful detector on plain prompts (normality test, Llama)",
              "normality_in_persona.py T3: StandardScaler + LR C=10 trained to separate benign from harmful PLAIN prompts "
              "(roleplay_check synthetic_v2; no jailbreak labels), no splits; folded HB AUC and mean folded transfer AUC on the "
              "jailbreak sets. The file's signed HB AUC field is dropped. Draw-based bases: mean of 5 draws.", ["hb_test"])
    src(f, "Detection + Persona share", S["label"] + " / normality direction share", "normality", "T3 → Detection (signed field dropped); T2.share → Persona share")
    DN = {"Persona traits": "PersonaJB", "Wikipedia acts (5)": "wiki_acts", "No Robots acts (5)": "nr_acts", "Random Gaussian (5)": "gauss", "Full 4096 activation": "raw"}
    for k, draws in d["T3"].items():
        rec("normality", "llama", DN[k], auc={"hb_test": mean([x[0] for x in draws])}, avg=mean([x[2] for x in draws]), draws=[x[2] for x in draws] if len(draws) > 1 else None)

def refdmu():
    S = setup("refdmu", "Refusal / Δμ direction per family, intact vs persona removed (in-sample)",
              "refdmu_tables.py A: score = h·v with Δμ fit on ALL rows (in-sample), folded AUC per family; HarmBench column copied "
              "from the 50-split test value of appendix_controls_data. The *_reversed flags (signed AUC < 0.5) are not shown. "
              "Avg = mean of the 5 families (mean5).", ["hb_test"] + FAM5)
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        f = base / "refdmu_tables.json"; d = load(f); src(f, "Detection", S["label"], "refdmu", "A block only; B block (d′, within-group AUC) not shown")
        for k, e in d["A"].items():
            dn, cn = k.split("|")
            rec("refdmu", m, "refusal" if dn == "refusal" else "jb_dir", cond="full" if cn == "intact" else "no_persona",
                auc={"hb_test": e["HarmBench"], **{s: e[F] for s, F in (("gcg", "GCG"), ("pair", "PAIR"), ("pap", "PAP"), ("gptfuzz", "GPTFuzz"), ("pez", "PEZ"))}}, avg=e["mean5"])

# ================================================================ STEERING
ST = []        # steering records
ABL = []
def tb(x): return x is True or x == "True" or x == "true"
def gzp(p):
    p = Path(p); return p if p.exists() else Path(str(p) + ".gz")
def exists_any(p): return Path(p).exists() or Path(str(p) + ".gz").exists()
def jl(p):
    p = gzp(p)
    with (gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)) as f: return [json.loads(l) for l in f if l.strip()]

NEG_POLE_GEMMA = {"stoic", "dispassionate", "literal", "dominant", "reserved", "confident"}
def arm_info(a, model):
    """-> (direction label, group, component variant)"""
    var = "full"
    for suf, v in (("_nopersona", "persona removed"), ("_personaonly", "persona component only"), ("_nonr", "No Robots span removed (NOT a valid control)")):
        if a.endswith(suf): a, var = a[: -len(suf)], v
    a = a.replace("__full", "")
    m = re.match(r"(?:alpaca1_)?alpacajb_d(\d)$", a)
    if m: return f"Alpaca-dictionary detector normal (draw {m[1]})", "control dictionary", var
    m = re.match(r"extra_(nr|wiki)jb_d(\d)$", a)
    if m: return f"{'No Robots' if m[1] == 'nr' else 'Wikipedia'}-dictionary detector normal (draw {m[2]})", "control dictionary", var
    m = re.match(r"(?:alpaca2_|extra_|nopersona_sel_)?sel_(acts|pca|nr|wiki)_(\d)$", a)
    if m: return {"acts": "Alpaca activation", "pca": "Alpaca PC", "nr": "No Robots activation", "wiki": "Wikipedia activation"}[m[1]] + f" #{m[2]} (top single direction)", "control single direction", var
    m = re.match(r"nopersona_det_alpacajb_d(\d)$", a)
    if m: return f"Alpaca-dictionary detector normal (draw {m[1]})", "control dictionary", var
    m = re.match(r"(random_full|random_persona|random_traitcombo|shuffled_persona)_s(\d)$", a)
    if m: return {"random_full": "Random direction (full space)", "random_persona": "Random direction in persona span", "random_traitcombo": "Random trait combination",
                  "shuffled_persona": "Shuffled PersonaJB direction"}[m[1]] + f" (seed {m[2]})", "random / null", var
    if a.startswith("trait__") or a in ("utilitarian", "edgy", "closure_seeking"):
        t = a.replace("trait__", ""); return f"Trait: {t}" + (" (−)" if model == "gemma" and t in NEG_POLE_GEMMA else ""), "single trait", var
    if a.startswith("paper:"): a = a[6:]
    return {"persona": ("PersonaJB direction", "main", var), "w_persona": ("PersonaJB direction", "main", var), "refusal": ("Refusal direction", "main", var),
            "refusal_pos1_neg": ("Refusal direction", "main", var), "dmu": ("Jailbreak direction Δμ", "main", var), "w_raw": ("Raw-probe normal", "main", var),
            "offdomain_gsm8k": ("Off-domain GSM8k direction", "random / null", var), "none": ("Unsteered", "random / null", var)}.get(a, (a, "other", var))

def st(m, bench, arm, src_id, snap, pool, dose, alpha, scoring, asr=None, lo=None, hi=None, n=None, deg=None, coh=None, coh_lo=None, coh_hi=None, diff=None, ci_kind=None, label=None, group=None, var=None, curve=None):
    lab, grp, v = arm_info(arm, m)
    r = {"m": m, "b": bench, "arm": arm, "dir": label or lab, "g": group or grp, "var": var or v, "src": src_id, "snap": snap, "pool": pool, "dose": dose,
         "a": alpha, "sc": scoring, "asr": num(asr), "lo": num(lo), "hi": num(hi), "n": n, "deg": num(deg), "coh": num(coh), "clo": num(coh_lo), "chi": num(coh_hi)}
    if diff: r["diff"] = diff
    if ci_kind: r["ck"] = ci_kind
    if curve: r["curve"] = curve
    ST.append({k: v for k, v in r.items() if v is not None})

SNAP = {"off": "Llama official meta-llama 0e9e39f2", "mir": "Llama mirror 247cb7c3 (no system header)", "unk": "Llama snapshot not stated in files",
        "gem": "Gemma-4-31B-it fb9ae262", "na": "n/a (attack, no steering)"}
STSRC = {}
def stsrc(id, label, path, protocol, snap):
    STSRC[id] = {"label": label, "path": rel(path), "protocol": protocol, "snap": snap}; src(path, "Steering", label, None, protocol[:120])

def per_gen(path, m, bench, arm, sid, snap, pool, dose, alpha, schedule=None):
    rows = jl(path); n = len(rows)
    jb = sum(tb(r.get("final_jailbroken")) for r in rows); dg = sum(tb(r.get("final_degenerate")) for r in rows)
    curve = None
    if schedule and len(schedule) > 1:
        curve = [[a, num(sum(1 for r in rows if tb(r.get("final_jailbroken")) and r.get("min_alpha_to_jailbreak") not in (None, "None") and float(r["min_alpha_to_jailbreak"]) <= a + 1e-9) / n)] for a in schedule]
    st(m, bench, arm, sid, snap, pool, dose, alpha, "harness", asr=jb / n, n=n, deg=dg / n, coh=jb / (n - dg) if n > dg else None, curve=curve)

def steering():
    NB = NSC / "steer_bench"
    stsrc("steer_bench", "steer_bench (official snapshot, refused-300 / JBB-100 / WildJB-298)", NB / "steer_nopersona_bench.json",
          "Llama L16, cumulative α 0.1→0.5 (each prompt stops at its first jailbreak), greedy, gpt-4.1-mini in-loop judge, 1 sample. "
          "HarmBench pool = 300 refused prompts (seed 0); JBB = 100 goals; WildJB = 298 refused of the first 600. 'harness' = "
          "judge label with word-5-gram degeneracy filter; 'corrected' = also fails jailbreaks whose first 300 chars repeat "
          "(char 10-gram > 0.5) or are < 40 chars. CI: behaviour-clustered bootstrap (2000), percentile. 'paper:' rows are "
          "copied from the paper cells: mirror snapshot, other pools (HB 1935 / WildJB 975 refused).", "off")
    d = load(NB / "steer_nopersona_bench.json")
    for k, e in d.items():
        sc, bench, arm = k.split("|")
        paper = e.get("paper")
        diff = {"v": num(e["diff"]), "lo": num(e["diff_lo"]), "hi": num(e["diff_hi"]), "vs": "same direction, full"} if "diff" in e else None
        st("llama", bench, arm, "steer_bench_paper" if paper else "steer_bench", "mir" if paper else "off",
           {"harmbench": "HB refused 1935", "jbb": "JBB 100", "wildjb": "WildJB refused 975"}[bench] if paper else {"harmbench": "HB refused-300 (seed 0)", "jbb": "JBB 100", "wildjb": "WildJB refused 298"}[bench],
           "cum", 0.5, sc, asr=e["asr"] / 100, lo=e["lo"] / 100, hi=e["hi"] / 100, n=e["n"], diff=diff and {k2: (v / 100 if isinstance(v, (int, float)) else v) for k2, v in diff.items()}, ci_kind="behaviour bootstrap")
    STSRC["steer_bench_paper"] = {"label": "paper refusal / Δμ cells (copied into steer_nopersona_bench.json)", "path": rel(NB / "steer_nopersona_bench.json"),
                                  "protocol": "Mirror snapshot runs on the paper pools; harness scoring (the 'corrected' copies are identical).", "snap": "mir"}
    have = {(r["b"], r["arm"], r["sc"]) for r in ST if r["src"] == "steer_bench"}
    p = load(NB / "personaonly_harmbench.json"); src(NB / "personaonly_harmbench.json", "Steering", "steer_bench persona-only arms", None, "asr for arms missing from steer_nopersona_bench; paired vs_full diffs")
    for arm, a in p["asr"].items():
        if ("harmbench", arm, "harness") in have: continue
        pv = (p["paired"].get(arm) or {}).get("vs_full")
        st("llama", "harmbench", arm, "steer_bench", "off", "HB refused-300 (seed 0)", "cum", 0.5, "harness", asr=a / 100, n=300,
           diff={"v": num(pv["diff"] / 100), "lo": num(pv["lo"] / 100), "hi": num(pv["hi"] / 100), "vs": "same direction, full"} if pv else None)
    # per-generation official arms without saved summary
    stsrc("steer_v2", "controls v2 run (official snapshot, HB refused-300)", NSC / "steer_v2_results",
          "Per-generation files (steer_fixed_direction.py), Llama L16, cumulative α 0.1→0.5, HB refused-300 seed 0, harness scoring. "
          "Rates re-aggregated here from the saved per-prompt labels: ASR = mean(final_jailbroken); degenerate = mean("
          "final_degenerate) (only never-flipped pairs can be degenerate); coherent ASR = jailbroken / non-degenerate. No CI saved. "
          "Arms whose ASR equals a steer_bench arm (same run) are dropped.", "off")
    sb = {(r["arm"]): r["asr"] for r in ST if r["src"] == "steer_bench" and r["b"] == "harmbench" and r["sc"] == "harness"}
    alias = {"alpaca1_alpacajb_d2": "alpacajb_d2", "alpaca2_sel_pca_1": "sel_pca_1", "alpaca2_sel_acts_1": "sel_acts_1", "nopersona_det_alpacajb_d2_nopersona": "alpacajb_d2_nopersona",
             "nopersona_sel_sel_acts_1_nopersona": "sel_acts_1_nopersona", "nopersona_sel_sel_pca_1_nopersona": "sel_pca_1_nopersona"}
    for f in sorted((NSC / "steer_v2_results").glob("*.jsonl*")):
        arm = f.name.replace(".jsonl.gz", "").replace(".jsonl", ""); rows = jl(f); asr = sum(tb(r["final_jailbroken"]) for r in rows) / len(rows)
        if arm in alias and alias[arm] in sb and abs(sb[alias[arm]] - round(asr, 5)) < 1e-4:
            continue
        per_gen(f, "llama", "harmbench", arm, "steer_v2", "off", "HB refused-300 (seed 0)", "cum", 0.5, [0.1, 0.2, 0.3, 0.4, 0.5])
    n = load(NSC / "nopersona_steer_results.json")
    stsrc("nopersona", "persona-removed steering pairs (corrected scoring, normal CI)", NSC / "nopersona_steer_results.json",
          "nopersona_steer_results.py: corrected ASR (char-degenerate rule) with vs without the persona component, HB refused-300, "
          "official snapshot; diff CI = paired per-prompt normal 95%. Only the No Robots row is kept here (the others duplicate "
          "steer_bench corrected rows, which carry bootstrap CIs). No Robots removal is NOT a valid non-persona control.", "off")
    for k, e in n.items():
        if "No Robots" not in k: continue
        st("llama", "harmbench", "alpacajb_d2_nonr", "nopersona", "off", "HB refused-300 (seed 0)", "cum", 0.5, "corrected", asr=e["removed"], n=e.get("n"),
           diff={"v": num(e["diff"]), "lo": num(e["ci"][0]), "hi": num(e["ci"][1]), "vs": "same direction, full"})
    # mirror runs
    for root, sid, lab, dose in ((FTO / "outputs_controls_cumulative", "ctrl_cum", "controls, cumulative (mirror snapshot)", "cum"),
                                 (FTO / "steer_fixed_directions_controls", "ctrl_fixed", "controls, single dose α=0.5 (mirror snapshot)", "single")):
        stsrc(sid, lab, root, "Per-generation results.jsonl + summary.json (config.model = mirror 247cb7c3), Llama L16, HB refused-300 seed 0, "
              + ("cumulative α 0.1→0.5" if dose == "cum" else "single dose α = 0.5") + ". Rates re-aggregated from per-prompt labels (harness scoring); no CI saved.", "mir")
        for dd in sorted((root / "harmbench").iterdir()):
            fr = dd / "results.jsonl"
            if not exists_any(fr): continue
            sm = load(dd / "summary.json"); snap = "mir" if "247cb7c3" in sm["config"]["model"] else ("off" if "0e9e39f2" in sm["config"]["model"] else "unk")
            sch = sm.get("alpha_schedule")
            per_gen(fr, "llama", "harmbench", dd.name, sid, snap, "HB refused-300 (seed 0)", dose, max(sch), sch if dose == "cum" else None)
    stsrc("alpha_sweep", "single-dose α sweep (mirror snapshot)", FTO / "outputs_alpha_sweep",
          "outputs_alpha_sweep/a{α}: single dose at each α (0.1..0.5), PersonaJB / refusal / Δμ, HB refused-300 seed 0, mirror snapshot "
          "(summary config). Rates re-aggregated from per-prompt labels; no CI saved.", "mir")
    for ad in sorted((FTO / "outputs_alpha_sweep").glob("a*")):
        a = float(ad.name[1:])
        for dd in sorted((ad / "harmbench").iterdir()):
            if not exists_any(dd / "results.jsonl"): continue
            sm = load(dd / "summary.json"); snap = "mir" if "247cb7c3" in sm["config"]["model"] else ("off" if "0e9e39f2" in sm["config"]["model"] else "unk")
            per_gen(dd / "results.jsonl", "llama", "harmbench", dd.name, "alpha_sweep", snap, "HB refused-300 (seed 0)", "single", a)
    # paper Llama cells
    R = PVJ / "results"
    pc = load(R / "steering_cell_cis.json")
    stsrc("paper_llama", "paper Llama cells (steering_cell_cis.json)", R / "steering_cell_cis.json",
          "Paper steering table cells: cumulative α (PersonaJB ran to 0.7 and was recut), harness scoring, pools HB refused 1935 (159 "
          "behaviours; classic attacks 159–400 test cases), JBB 100, WildJB 2000 (all prompts, not refused-only). CI: "
          "behaviour-clustered where 'clustered', else per pair. SNAPSHOT: refusal / Δμ = mirror 247cb7c3 (plot_steer_bench.py "
          "comment); PersonaJB and single-trait arms = Dominic's runs, not stated in the local files (memory says official). "
          "Classic attacks = no steering.", "unk")
    for k, e in pc.items():
        arm, bench = k.split("|")
        atk = arm in ("GCG", "PAIR", "PAP", "GPTFuzz", "PEZ")
        snap = "na" if atk else ("mir" if arm in ("refusal", "dmu") else "unk")
        st("llama", bench, arm, "paper_llama", snap, {"harmbench": "HB refused 1935" if not atk else "HB test cases", "jbb": "JBB 100", "wildjb": "WildJB 2000 (all)"}[bench],
           "attack" if atk else "cum", None if atk else 0.5, "harness", asr=e["asr"], lo=e["lo"], hi=e["hi"], n=e.get("n_pairs"),
           ci_kind="behaviour-clustered bootstrap" if e.get("clustered") else "per-pair", label=f"Attack: {arm} (no steering)" if atk else None, group="classic attack" if atk else None)
    w = load(R / "wjb_refused_cells.json")
    stsrc("paper_llama_wjbref", "paper Llama cells, WildJB refused-only (wjb_refused_cells.json)", R / "wjb_refused_cells.json",
          "Same runs as the paper cells, restricted to the 975 (traits: 200) WildJB prompts the unsteered model refused; CI as saved.", "unk")
    for arm, e in w.items():
        atk = arm in ("GCG", "PAIR", "PAP", "GPTFuzz", "PEZ")
        st("llama", "wildjb_refused", arm, "paper_llama_wjbref", "na" if atk else ("mir" if arm in ("refusal", "dmu") else "unk"), f"WildJB refused {e['n_refused']}",
           "attack" if atk else "cum", None if atk else 0.5, "harness", asr=e["asr_refused"], lo=e["lo"], hi=e["hi"], n=e["n_refused"],
           label=f"Attack: {arm} (no steering)" if atk else None, group="classic attack" if atk else None)
    # Gemma
    g = load(R / "gemma_steering_cell_cis.json")
    stsrc("gemma_cells", "Gemma steering cells, gibberish-corrected (gemma_steering_cell_cis.json)", R / "gemma_steering_cell_cis.json",
          "extract_gemma_steering_cells.py: Gemma-4-31B-it L30, cumulative α 0.1→0.5, gpt-4.1-mini judge; jailbreaks failing the "
          "looks_like_text guard are relabelled degenerate (these are the VALID Gemma cells; raw summary.json ASRs are superseded). "
          "ASR = jailbroken / all; coherent ASR = jailbroken / non-degenerate; degenerate = n_degenerate / n. CI: percentile "
          "bootstrap, behaviour-clustered for HarmBench (10k), per pair for JBB (10k) / WildJB (2k). Pools: HB refused 3117, JBB "
          "100, WildJB 2000, WildJB refused 1816. Traits marked (−) are steered on their negative pole.", "gem")
    for k, e in g.items():
        arm, bench = k.split("|")
        np_ = e.get("n_pairs")
        st("gemma", bench, arm, "gemma_cells", "gem", {"harmbench": "HB refused 3117", "jbb": "JBB 100", "wildjb": "WildJB 2000 (all)", "wildjb_refused": "WildJB refused 1816"}[bench],
           "cum", 0.5, "looks_like_text corrected", asr=e["asr"], lo=e["lo"], hi=e["hi"], n=np_, deg=e["n_degenerate"] / np_ if np_ and e.get("n_degenerate") is not None else None,
           coh=e.get("asr_coherent"), coh_lo=e.get("coherent_lo"), coh_hi=e.get("coherent_hi"), ci_kind="bootstrap")
    c = load(R / "gemma_steering_cell_cis_classic.json")
    stsrc("gemma_classic", "Gemma classic attacks (no steering)", R / "gemma_steering_cell_cis_classic.json", "Attack ASR on HarmBench test cases, CI as saved (bootstrap).", "na")
    for k, e in c.items():
        arm, bench = k.split("|")
        st("gemma", bench, arm, "gemma_classic", "na", "HB test cases", "attack", None, "harness", asr=e["asr"], lo=e["lo"], hi=e["hi"], n=e["n_pairs"],
           label=f"Attack: {arm} (no steering)", group="classic attack")
    gs = load(NSC / "gemma/steer/steer_split_gemma.json")
    stsrc("gemma_split", "Gemma persona-component steering (HB refused-300)", NSC / "gemma/steer/steer_split_gemma.json",
          "norm_share_controls/gemma/steer (plot_steer_gemma.py): HB refused-300 (pool seed 0 of the 3117), cumulative α 0.1→0.5, "
          "coherent ASR (judged jailbroken, not degenerate, passes looks_like_text), percent → fraction; CI and paired diffs = "
          "behaviour bootstrap (2000). 'earlier run' rows are the main Gemma cells restricted to these 300 prompts.", "gem")
    for k, e in gs.items():
        parts = k.split("|"); grp, lab = parts[0], parts[1] if len(parts) > 1 else ""
        base = {"PersonaJB direction": "PersonaJB direction", "Refusal direction": "Refusal direction", "Jailbreak direction": "Jailbreak direction Δμ", "Random direction": "Random direction (full space)"}.get(grp, grp)
        var = {"persona component only": "persona component only", "persona component removed": "persona removed", "original": "full", "detector normal": "full",
               "activation pattern, persona space": "activation pattern in persona space"}.get(lab, lab or "full")
        diff = {"v": num(e["diff"] / 100), "lo": num(e["diff_lo"] / 100), "hi": num(e["diff_hi"] / 100), "vs": e.get("vs", "")} if "diff" in e else None
        st("gemma", "harmbench", k, "gemma_split", "gem", "HB refused-300 (seed 0)" + (" · earlier run" if e.get("earlier_run") else ""), "cum", 0.5,
           "looks_like_text corrected", asr=e["asr"] / 100, lo=e["lo"] / 100, hi=e["hi"] / 100, n=e["n"], diff=diff, label=base,
           group="main" if grp != "Random direction" else "random / null", var=var)
    # ablation
    a = load(R / "ablation_cells.json"); src(R / "ablation_cells.json", "Steering", "Directional ablation (Llama)", None, "refused / still-jailbroken / degenerate rates with CIs as saved")
    for k, e in a.items():
        arm, bench = k.split("|"); at16 = arm.endswith("@16"); base = arm.replace("@16", "")
        lab, grp, _ = arm_info(base, "llama")
        g2 = e.get("refused_gain_vs_none") or {}
        ABL.append({"m": "llama", "b": bench, "arm": arm, "dir": ("No ablation" if base == "none" else lab) + (" (layer 16 only)" if at16 else " (all layers)"), "g": grp,
                    "n": e.get("n"), "n_scored": e.get("n_scored"), "refused": [num(e["refused"]["rate"]), num(e["refused"]["lo"]), num(e["refused"]["hi"])],
                    "still": [num(e["still_jailbroken"]["rate"]), num(e["still_jailbroken"]["lo"]), num(e["still_jailbroken"]["hi"])] if e.get("still_jailbroken") else None,
                    "deg": [num(e["degenerate"]["rate"]), num(e["degenerate"]["lo"]), num(e["degenerate"]["hi"])],
                    "gain": [num(g2.get("delta")), num(g2.get("lo")), num(g2.get("hi"))] if g2 else None, "parse_fail": e.get("parse_failures")})

# ================================================================ PERSONA SHARE (generic panels)
PANELS = []
def panel(id, title, protocol, cols, rows, fmt="share", chart=True, note=None, warn=None):
    PANELS.append({"id": id, "title": title, "protocol": protocol, "cols": cols, "rows": rows, "fmt": fmt, "chart": chart, "note": note, "warn": warn})

GEMMA_WARN = ("Gemma caveat: Gemma activations carry 3 massive dimensions (2041, 3970, 1451; cosine-null audit), which inflate "
              "shares of activation-derived vectors (e.g. Wikipedia 0.92 in the persona span). The random-span null is 240/5376 ≈ "
              "0.045 on Gemma vs 229/4096 ≈ 0.056 on Llama.")
def share_tab():
    SP = [("persona", "Persona span"), ("alpaca_pca", "Alpaca PCA-K"), ("alpaca_acts", "Alpaca acts"), ("norobots_acts", "No Robots acts"), ("wiki_acts", "Wikipedia acts"), ("random", "Random Gaussian")]
    rows, rows2 = [], []
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        d = load(base / "appendix_controls_data.json")
        for t, tl in (("refusal", "Refusal direction"), ("dmu", "Jailbreak direction Δμ")):
            rows.append({"m": m, "label": tl, "v": {k: [num(d["share"][t][k]["mean"]), num(d["share"][t][k]["min"]), num(d["share"][t][k]["max"])] for k, _ in SP if k in d["share"][t]}})
        NL = [("refusal", "Refusal (observed)"), ("strat_refusal", "Stratified-shuffle refusal (clean null)"), ("shuf_refusal", "Pooled-shuffle refusal (inflated null)"),
              ("dmu", "Δμ (observed)"), ("strat_dmu", "Stratified-shuffle Δμ (clean null)"), ("shuf_dmu", "Pooled-shuffle Δμ (inflated null)"),
              ("shuf_alpaca", "Shuffled Alpaca contrasts"), ("random_direction", "Random Gaussian direction")]
        for k, l in NL:
            e = d["share_persona"].get(k)
            if e: rows2.append({"m": m, "label": l, "v": {"mean": num(e["mean"]), "p5": num(e["p5"]), "p95": num(e["p95"]), "n": e["n"]}})
    panel("dir_spans", "How much of the refusal / jailbreak direction lies in each span",
          "appendix_controls_data.py share block: ||Q_Sᵀv||² / ||v||², Q_S orthonormal basis of a K-dim span (K = 229 Llama / 240 Gemma). "
          "Llama L16 / Gemma L30, last token. Activation spans = K random activations of that corpus, mean [min, max] over 5 draws "
          "(persona and Alpaca PCA: one basis).", [{"k": k, "l": l} for k, l in SP], rows, warn=GEMMA_WARN)
    panel("persona_nulls", "Persona-span share of the directions vs null directions",
          "appendix_controls_data.py share_persona: share of each direction in the persona span; null distributions over n draws "
          "(mean, 5th / 95th percentile). Stratified shuffles are the clean null; pooled shuffles leak the real direction (inflated).",
          [{"k": "mean", "l": "mean"}, {"k": "p5", "l": "p5"}, {"k": "p95", "l": "p95"}, {"k": "n", "l": "n draws"}], rows2, chart=False)
    rows = []
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        d = load(base / "w_out_in_spans.json")
        spans = list(d["share"])
        for dirn in ("w_in", "w_out", "jailbreak direction", "refusal direction", "refusal, persona removed"):
            rows.append({"m": m, "label": {"w_in": "w_in (Δμ persona component)", "w_out": "w_out (Δμ persona removed)"}.get(dirn, dirn),
                         "v": {re.sub(r"-(229|240)|\((1145|1200)\)", "K", s).replace("PCAK", "PCA-K"): num(d["share"][s][dirn]) for s in spans}})
    cols = []
    for r in rows:
        for k in r["v"]:
            if k not in [c["k"] for c in cols]: cols.append({"k": k, "l": k})
    panel("w_out", "What is left of Δμ outside the persona span (w_out) — share in other spans",
          "w_out_in_spans.py: share of w_in / w_out / Δμ / refusal in each span. Activation spans = draw 0; '− P' = draw 4 with the "
          "persona component removed; 'x5' = 5 draws stacked. Llama K=229 (1145 for x5) / Gemma K=240 (1200).", cols, rows, warn=GEMMA_WARN)
    rows = []
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        d = load(base / "remove_nonpersona_span.json")
        for sn, s in d["spans"].items():
            rows.append({"m": m, "label": sn + ("" if sn != "No Robots - P" else " (not a removal control)"), "v": {"var_share": num(s["var_share"]), "dmu_share": num(s["dmu_share"]), "overlap": num(s["overlap_with_persona"])}})
    panel("spans_hb", "Removal spans: share of HarmBench variance and of Δμ, overlap with persona span",
          "remove_nonpersona_span.py: for each removal span, share of the total (centered) variance of the HB activations, share "
          "of ||Δμ||², and mean overlap with the persona span. '− P' = activations with their persona component removed.",
          [{"k": "var_share", "l": "HB variance share"}, {"k": "dmu_share", "l": "Δμ share"}, {"k": "overlap", "l": "overlap w/ persona"}], rows, warn=GEMMA_WARN)
    rows, pts = [], []
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        d = load(base / "persona_part_spans.json")
        for k, e in d.items():
            if k == "p95": continue
            sp, so = k.split("|")
            rows.append({"m": m, "label": f"{so} vectors in {sp} span" + (" (NOT a valid persona control)" if sp.startswith("No Robots") else ""),
                         "v": {"share": num(e["share"]), "xf": num(e["xf"]), "xf_in": num(e["xf_in"]), "xf_out": num(e["xf_out"]), "n_in_ge": e["n_in_ge"], "n_out_le": e["n_out_le"], "p95": num(d["p95"])}})
        pv = load(base / "persona_part_pervector.json")
        for k, e in pv.items():
            if not isinstance(e, dict): continue
            so, sp = k.split("|"); idx = e["top_idx"]
            pts.append({"m": m, "src": so, "span": re.sub(r"-(229|240)", "-K", sp), "x": [num(e["xf_v"][i]) for i in idx], "yin": [num(e["xf_in"][i]) for i in idx],
                        "yout": [num(e["xf_out"][i]) for i in idx], "p95": num(pv["p95"]), "floor": num(pv["floor"])})
    panel("persona_part", "Persona part of ordinary activations: share and where their transfer signal sits",
          "persona_part_spans.py: each Wikipedia / No Robots / Alpaca activation v split into v_in (in the span) and v_out. share = "
          "median ||v_in||² over all vectors; xf / xf_in / xf_out = mean folded transfer AUC of h·v, h·v_in, h·v_out over the top-50 "
          "vectors by HB AUC (single vectors, nothing fitted); n_in_ge = # of 50 with v_in ≥ v, n_out_le = # with v_out ≤ the "
          "random-direction 95th pct (p95).", [{"k": "share", "l": "share in span"}, {"k": "xf", "l": "xf(v)"}, {"k": "xf_in", "l": "xf(v_in)"},
          {"k": "xf_out", "l": "xf(v_out)"}, {"k": "n_in_ge", "l": "#v_in≥v /50"}, {"k": "n_out_le", "l": "#v_out≤p95 /50"}, {"k": "p95", "l": "noise p95"}], rows, fmt="mixed", chart=False, warn=GEMMA_WARN)
    PERVEC.extend(pts)
    d = load(NSC / "pcs_in_persona.json"); rows = []; curves = []
    for pn, pl in (("alpaca_pca", "Alpaca PCs"), ("hb_pca", "HarmBench PCs")):
        v = {}
        for sp in ("Persona traits", "Wikipedia acts", "No Robots acts", "Random Gaussian"):
            v[sp] = num(mean(d[pn][sp])); curves.append({"m": "llama", "pcs": pl, "span": sp, "y": [num(x) for x in d[pn][sp]]})
        rows.append({"m": "llama", "label": f"{pl} (229), mean per-PC share", "v": v})
    panel("pcs_in_persona", "Share of each Alpaca / HarmBench PC in the persona span vs control spans (Llama)",
          "pcs_in_persona.py: per PC (229 top PCs), share of its squared norm in each span (control spans: mean over 5 draws). Table = "
          "mean over the 229 PCs (re-aggregated); chart = share per PC index.", [{"k": s, "l": s} for s in ("Persona traits", "Wikipedia acts", "No Robots acts", "Random Gaussian")], rows)
    PCCURVES.extend(curves)
    d = load(NSC / "normality_in_persona.json")
    panel("normality", "Normality direction n = mean(benign) − mean(harmful) plain prompts: share in spans (Llama)",
          "normality_in_persona.py T2: n from roleplay_check synthetic_v2 plain prompts (no jailbreak labels); share of ||n||² per span "
          "(control spans: mean of 5 draws). Folded HB AUC of h·n (in-sample) = " + f"{d['T2']['auc_hb']:.3f}.",
          [{"k": k, "l": k} for k in d["T2"]["share"]], [{"m": "llama", "label": "normality direction n", "v": {k: num(v) for k, v in d["T2"]["share"].items()}}])
    r = load(NSC / "results.json"); rows = []
    for t in ("refusal", "dmu", "refusal_rebuilt"):
        e = r["A_subspaces"][t]
        rows.append({"m": "llama", "label": {"refusal": "Refusal (Arditi)", "dmu": "Δμ", "refusal_rebuilt": "Refusal rebuilt (harmful − Alpaca)"}[t],
                     "v": {"persona": num(e["persona"]), "alpaca_pca229": num(e["alpaca_pca229"]), "alpaca_acts229": num(e["alpaca_acts229"]["mean"]),
                           "alpaca_acts229_p95": num(e["alpaca_acts229"]["p95"]), "random229": num(e["random229"]["mean"]), "random229_p95": num(e["random229"]["p95"])}})
    B = r["B_directions_in_persona"]
    for k in ("alpaca_mean", "hb_mean"):
        rows.append({"m": "llama", "label": {"alpaca_mean": "Alpaca mean activation", "hb_mean": "HarmBench mean activation"}[k], "v": {"persona": num(B[k])}})
    rows.append({"m": "llama", "label": "Alpaca activations: share of centered variance", "v": {"persona": num(r["C_alpaca_variance_share"]["persona"]),
                 "alpaca_pca229": num(r["C_alpaca_variance_share"]["alpaca_pca229"]), "random229": num(r["C_alpaca_variance_share"]["random229"]["mean"])}})
    panel("first_controls", "First norm-share controls (Llama, norm_share_controls.py)",
          "results.json: share in the persona span vs Alpaca PCA-229, span of 229 random Alpaca activations (mean / p95 over 200 "
          "draws) and random 229-dim spans (mean / p95 over 200). The pooled-shuffle nulls of this file (B.shuffled_*) are "
          "superseded by the stratified nulls above and not shown.",
          [{"k": "persona", "l": "persona"}, {"k": "alpaca_pca229", "l": "Alpaca PCA-229"}, {"k": "alpaca_acts229", "l": "Alpaca acts (mean)"},
           {"k": "alpaca_acts229_p95", "l": "Alpaca acts p95"}, {"k": "random229", "l": "random (mean)"}, {"k": "random229_p95", "l": "random p95"}], rows)
    rows = []
    for m, base in (("llama", NSC), ("gemma", NSC / "gemma")):
        d = load(base / "what_is_dmu_out.json")
        for k in ("dmu intact", "persona component (w_in)", "persona removed (w_out)"):
            e = d[k]; rows.append({"m": m, "label": k, "v": {kk: num(fold(e[kk]) if "auc" in kk or kk in ("harmful_vs_benign", "roleplay_vs_plain") else e[kk]) for kk in e}})
    panel("dmu_out", "What is Δμ's remainder? Template / behaviour structure of Δμ, w_in, w_out",
          "what_is_dmu_out.py: R² of h·v explained by template / behaviour / both; folded in-sample AUCs (jailbroken vs refused, within "
          "template, within behaviour, harmful vs benign, role-play vs plain); signed cosines with −refusal and with the normality "
          "direction. Not a share result; context for w_out.",
          [{"k": k, "l": k} for k in ("r2_template", "r2_behaviour", "r2_both", "auc", "auc_within_template", "auc_within_behaviour", "harmful_vs_benign", "roleplay_vs_plain", "cos_refusal", "cos_normality")],
          rows, fmt="raw", chart=False)
    y = load(NSC / "why_dmu_no_drop.json")
    panel("top10pc", "Share of the top-10 HarmBench PCs in the persona span (Llama)",
          "why_dmu_no_drop.py top10_pc_share_in_persona: per HB PC (1..10), share of its squared norm in the persona span.",
          [{"k": f"PC{i+1}", "l": f"PC{i+1}"} for i in range(10)], [{"m": "llama", "label": "HB PC share in persona span", "v": {f"PC{i+1}": num(x) for i, x in enumerate(y["top10_pc_share_in_persona"])}}])
    rows = []
    for m in ("llama", "gemma"):
        f = RELROOT / f"derived/{m}/analysis/persona_span.json"; d = load(f); src(f, "Persona share", "Clean-repo release persona span", None)
        rows.append({"m": m, "label": "release persona_span.json", "v": {"refusal": num(d["share"]["refusal"]), "jailbreak": num(d["share"]["jailbreak"]), "chance": num(d["chance"]),
                     **{f"auc {k}": num(fold(v)) for k, v in d["auc"].items()}}})
    panel("release_span", "Clean-repo release: persona-span share and AUC of the persona / remainder parts",
          "~/persona-jailbreaks 5_analysis.py on the release data: share of refusal / jailbreak direction in the persona span (chance = "
          "K/d); folded AUC of the full direction, its persona component and the remainder (unfitted, full pool).",
          [{"k": k, "l": k} for k in ["refusal", "jailbreak", "chance", "auc refusal/full", "auc refusal/persona", "auc refusal/remainder", "auc jailbreak/full", "auc jailbreak/persona", "auc jailbreak/remainder"]],
          rows, fmt="raw", chart=False)
    for f in ("results.json", "w_out_in_spans.json", "pcs_in_persona.json", "persona_part_spans.json", "persona_part_pervector.json", "what_is_dmu_out.json", "why_dmu_no_drop.json", "remove_nonpersona_span.json"):
        for base in (NSC, NSC / "gemma"):
            if (base / f).exists() and f != "remove_nonpersona_span.json": src(base / f, "Persona share", f.replace(".json", ""), None)
PERVEC, PCCURVES = [], []

def det_group(d):
    if d in DETS: return DETS[d][1]
    for pre, g in (("tpc_", "trait-matrix PCs"), ("lda_", "persona-LDA"), ("pcak_", "activation PCA(k)"), ("pm_", "probe methods"), ("PCA_", "trait-matrix PCs"),
                   ("top_", "top-k / random-k traits"), ("rand_", "top-k / random-k traits")):
        if d.startswith(pre): return g
    return "other"
_ORDER = list(DETS)
def det_order(d):
    m = re.match(r"(.*?)_?(\d+)$", d)
    if d in _ORDER: return (0, _ORDER.index(d), 0, d)
    if m: return (1, 0, m[1], int(m[2]))
    return (2, 0, d, 0)

# ================================================================ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke-q1"); ap.add_argument("--out", default=str(HERE / "data.json")); a = ap.parse_args()
    q1(a.smoke_q1); hb_select(); single_set(); dom_compare(); olmo_layer_sweep(); paper50(); removal50(); removal10(); dir50(); few_dirs()
    single_vec(); normality_det(); refdmu(); release(); qwen_sources(); pca_k(); complement(); steering(); share_tab()
    dets = sorted({r["d"] for r in RECS} | {h["d"] for h in HEAT})
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": MODELS, "sets": SETS,
           "dets": {d: {"label": det_label(d), "group": det_group(d), "o": i} for i, d in enumerate(sorted(dets, key=det_order))}, "conds": CONDS,
           "detection": {"setups": SETUPS, "records": RECS, "heat": HEAT, "complement": COMPLEMENT},
           "steering": {"records": ST, "sources": STSRC, "snaps": SNAP, "ablation": ABL},
           "share": {"panels": PANELS, "q1": SHARE_Q1, "pervec": PERVEC, "pccurves": PCCURVES},
           "sources": SOURCES}
    s = json.dumps(out, separators=(",", ":"), allow_nan=False)
    Path(a.out).write_text(s)
    print(f"wrote {a.out}: {len(s)/1e6:.2f} MB, {len(RECS)} detection records, {len(HEAT)} heatmaps, {len(ST)} steering records, {len(PANELS)} share panels, {len(SOURCES)} sources")
    if not SHARE_Q1: print("q1: PENDING (no q1_*.json in q1_detection/out)")

if __name__ == "__main__":
    main()
