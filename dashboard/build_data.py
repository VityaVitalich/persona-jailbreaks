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
    "no_random": "Random K-dim span removed (null)",
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

# ================================================================ DETECTION (q1 protocol only)
Q1_MODELS = ["llama", "olmo", "qwen", "gemma"]
Q1_TRAINS = {"hb": "HB human jailbreaks", "hbwjb": "HB + WJB pooled"}
Q1_PROTOCOL = ("q1 protocol. Each detector's readout (layer × tail position × C) is tuned by 5-fold cross-validation on the "
               "training set (folds grouped by HB template; WJB rows by behaviour), the selected cell is refit on all training "
               "rows and scored on the transfer sets. Folded AUC max(a, 1−a) per set. Avg = mean over PAP, GCG, PAIR, GPTFuzz, "
               "PEZ (avg_sets); WJB is an extra column when it is a test set (HB-only training) and is not averaged. CI = "
               "HarmBench-behaviour cluster bootstrap (B=1000); Δ CI = paired bootstrap. Components: persona span (K trait vectors "
               "of that layer) projected out / random K-dim Gaussian span projected out / only the persona component kept. "
               "Removal BEFORE training (q1_sweep.py): every activation the detector sees, including the Alpaca / Wikipedia / "
               "refusal inputs it is built from, is projected, and the detector is retrained; readout 'same as full' keeps the "
               "full detector's layer and position with C re-tuned, 're-selected' re-runs the whole selection. Removal AFTER "
               "training (q1_posthoc.py): the detector trained on full activations scores persona-removed test activations; "
               "its readout is the full detector's. Gaussian = mean of 3 draws.")
Q1_STATUS = []

def q1():
    outdir = FTO / "q1_detection/out"
    src(FTO / "q1_detection/q1_sweep.py", "Detection", "q1 sweep (removal before training): writer", None, "docstring = protocol")
    src(FTO / "q1_detection/q1_posthoc.py", "Detection", "q1 post-hoc (removal after training): writer", None, "docstring = protocol")
    for m in Q1_MODELS:
        for t in Q1_TRAINS:
            for tm, fn in (("before", f"q1_{m}_{t}.json"), ("after", f"q1_posthoc_{m}_{t}.json")):
                f = outdir / fn
                Q1_STATUS.append({"m": m, "t": t, "tm": tm, "file": rel(f), "ready": f.exists()})
                if not f.exists(): continue
                src(f, "Detection" + (" + Persona share" if tm == "before" else ""), f"q1 {'sweep' if tm == 'before' else 'post-hoc'} {MODELS[m]} {Q1_TRAINS[t]}", None)
                (q1_before if tm == "before" else q1_after)(load(f), m, t, f)

def _gauss_mean(entries, key="test_auc"):
    sets = set().union(*[e[key].keys() for e in entries])
    return {s: mean([e[key].get(s) for e in entries]) for s in sets}

def q1_before(d, m, t, f):
    AVG = d.get("avg_sets") or [s for s in d["test"] if s != "wjb"]
    TAIL = d["tail_tokens"]; sel = d["selected"]; dl = d["delta"]
    st = next(x for x in Q1_STATUS if x["m"] == m and x["t"] == t and x["tm"] == "before")
    st.update({"n_train": d.get("n_train"), "n_pos_train": d.get("n_pos_train"), "n_test": d.get("n_test"), "grid": d.get("grid"), "k": d.get("k"),
               "test": d["test"], "avg_sets": AVG, "swept_layers": d.get("swept_layers")})
    def one(det, cond, mode, e, dkey):
        dd = None
        if cond != "full" and dkey in dl:
            x = dl[dkey]; dd = {"full": {"v": num(x["delta"]), "ci": ci2(x["ci95"])}}
        return rec("q1", m, det, auc=e.get("test_auc"), avg=e.get("avg"), ci=e.get("avg_ci95"), cond=cond, train=t, readout=mode,
                   val=e.get("val_auc"), sel={"L": e.get("layer"), "p": e.get("position"), "tok": e.get("token"), "C": e.get("C")} if "layer" in e else None, dl=dd)
    dets = sorted({k.split("__")[0] for k in sel})
    for det in [x for x in dets if not x.startswith("gauss_d")]:
        for cond in ("full", "no_persona", "no_random", "only_persona"):
            for mode in ("fixed", "reselect"):
                k = f"{det}__{cond}__{'reselect' if cond == 'full' else mode}"
                if k in sel: r = one(det, cond, mode, sel[k], f"{det}__{cond}__{mode}"); r["tm"] = "before"
    draws = sorted(x for x in dets if x.startswith("gauss_d"))
    for cond in ("full", "no_persona", "no_random", "only_persona"):
        for mode in ("fixed", "reselect"):
            es = [sel[k] for k in (f"{g}__{cond}__{'reselect' if cond == 'full' else mode}" for g in draws) if k in sel]
            if not es: continue
            if cond == "full":
                sm = sel.get("gauss__full__summary", {}); avg, ci, dd = sm.get("avg"), sm.get("avg_ci95"), None
            else:
                x = dl.get(f"gauss__{cond}__{mode}"); avg = x["avg"] if x else mean([e["avg"] for e in es]); ci = None
                dd = {"full": {"v": num(x["delta"]), "ci": ci2(x["ci95"])}} if x else None
            r = rec("q1", m, "gauss", auc=_gauss_mean(es), avg=avg, ci=ci, cond=cond, train=t, readout=mode, val=mean([e["val_auc"] for e in es]),
                    sel={"per_draw": [f"L{e['layer']} p{e['position']} C={e['C']}" for e in es]}, dl=dd, draws=[e["avg"] for e in es]); r["tm"] = "before"
    groups = {}
    for k, v in d["cells"].items():
        det, cond, Ls, ps = k.split("__"); det = "gauss" if det.startswith("gauss_d") else det
        groups.setdefault((det, cond), {}).setdefault((int(Ls[1:]), int(ps[1:])), []).append(v)
    L = d.get("swept_layers") or d["layers"]; P = list(range(len(TAIL)))
    for (det, cond), g in groups.items():
        val = [[num(mean([c["val_auc"] for c in g[(l, p)]])) if (l, p) in g else None for p in P] for l in L]
        avg = [[num(mean([mean([fold(c["test_auc"][s]) for s in AVG]) for c in g[(l, p)]])) if (l, p) in g else None for p in P] for l in L]
        HEAT.append({"s": "q1", "m": m, "t": t, "d": det, "c": cond, "layers": L, "toks": TAIL, "metrics": {"val": val, "avg": avg}})
    SHARE_Q1.append({"model": m, "train": t, "info": d["info"], "layers": L, "toks": TAIL, "smoke": False, "file": rel(f)})

def q1_after(d, m, t, f):
    st = next(x for x in Q1_STATUS if x["m"] == m and x["t"] == t and x["tm"] == "after"); st.update({"test": d["test"], "avg_sets": d.get("avg_sets")})
    D_ = d["detectors"]
    def put(det, es):
        for cond in ("full", "no_persona", "no_random", "only_persona"):
            xs = [e[cond] for e in es if cond in e]
            if not xs: continue
            one = len(xs) == 1
            dd = None
            if cond != "full":
                dv = mean([x.get("delta") for x in xs])
                dd = {"full": {"v": num(dv), "ci": ci2(xs[0].get("delta_ci95")) if one else None}}
            sel = {"L": es[0]["layer"], "p": es[0]["position"], "C": es[0]["C"]} if one else {"per_draw": [f"L{e['layer']} p{e['position']} C={e['C']}" for e in es]}
            sel.update({"w_share_persona": num(mean([e.get("w_share_persona") for e in es])), "w_share_random": num(mean([e.get("w_share_random") for e in es]))})
            r = rec("q1", m, det, auc=_gauss_mean(xs) if not one else xs[0]["test_auc"], avg=mean([x["avg"] for x in xs]), ci=xs[0].get("avg_ci95") if one else None,
                    cond=cond, train=t, readout="fixed", sel=sel, dl=dd, draws=[x["avg"] for x in xs] if not one else None)
            r["tm"] = "after"
            W_SHARE.append({"m": m, "t": t, "d": det, "persona": sel["w_share_persona"], "random": sel["w_share_random"]}) if cond == "full" else None
    for det in [x for x in D_ if not x.startswith("gauss_d")]: put(det, [D_[det]])
    g = [D_[x] for x in sorted(D_) if x.startswith("gauss_d")]
    if g: put("gauss", g)
W_SHARE = []

SHARE_Q1 = []

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
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "data.json")); a = ap.parse_args()
    q1(); steering(); share_tab()
    dets = sorted({r["d"] for r in RECS} | {h["d"] for h in HEAT})
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": MODELS, "sets": SETS,
           "dets": {d: {"label": det_label(d), "group": det_group(d), "o": i} for i, d in enumerate(sorted(dets, key=det_order))}, "conds": CONDS,
           "detection": {"protocol": Q1_PROTOCOL, "models": Q1_MODELS, "trains": Q1_TRAINS, "status": Q1_STATUS, "records": RECS, "heat": HEAT, "w_share": W_SHARE},
           "steering": {"records": ST, "sources": STSRC, "snaps": SNAP, "ablation": ABL},
           "share": {"panels": PANELS, "q1": SHARE_Q1, "pervec": PERVEC, "pccurves": PCCURVES},
           "sources": SOURCES}
    s = json.dumps(out, separators=(",", ":"), allow_nan=False)
    Path(a.out).write_text(s)
    ready = sum(x["ready"] for x in Q1_STATUS)
    print(f"wrote {a.out}: {len(s)/1e6:.2f} MB, q1 files ready {ready}/{len(Q1_STATUS)}, {len(RECS)} detection records, {len(HEAT)} heatmaps, "
          f"{len(ST)} steering records, {len(PANELS)} share panels, {len(SOURCES)} sources")

if __name__ == "__main__":
    main()
