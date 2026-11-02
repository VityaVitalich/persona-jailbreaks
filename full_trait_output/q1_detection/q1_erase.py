"""Persona erasure test (detection only, CPU). Does the raw jailbreak probe still transfer once persona information is
LINEARLY UNRECOVERABLE, and is any drop more than a non-targeted edit of the same size causes?
Protocol = q1 (loaders verbatim from q1_sweep.py via q1_reencode.py): train HB human_jailbreak + WJB, 5-fold GroupKFold,
same C grid, transfer sets, folded AUC, HarmBench-behaviour cluster bootstrap (B=1000, seed 12345); cells = the raw probe's
and PersonaJB's q1-selected (layer, position).
X activations, Q = QR basis of that layer's K trait vectors, Z = X Q (persona coordinates of the ORIGINAL X), mu/Sigma on train.
Edits (fit on train, applied to train and transfer sets):
  span        X - X Q Q^T                                         (the usual "persona removed"; K dims)
  orth        span, then project out col(Cov(X_span, Z))           (<= 2K dims, orthogonal; Cov(X', Z) = 0 on train)
  leace       LEACE (Belrose et al. 2023) on X w.r.t. Z:  x' = x - S^{1/2} U U^T S^{-1/2} (x - mu),
              U = orth(S^{1/2} Q) (S = Sigma restricted to its range)  -> Cov(X', Z) = 0, minimal edit, rank <= K
Controls, one per edit and draw (3): RANK-MATCHED UNIFORM random directions - the same number of isotropic Gaussian
directions as the edit (seeds 4000+g), projected out of the raw activations; they know nothing about the data.
Per edit/control: (1) persona R^2 (ridge X' -> standardized Z, CV on train + transfer), (2) jailbreak raw probe retrained ->
transfer AUC + CI + paired delta vs full, (3) SPECIFICITY probe: HarmBench semantic category (6 classes, HB train rows only,
grouped CV by behaviour) -> macro one-vs-rest AUC on the transfer sets (copyright excluded); (4) GENERAL probes on standard
NLP tasks - SST-2 sentiment, AG News topic, cities true/false (edits fit on jailbreak data, applied there) - is general
information intact?
Light score test (no edit): regress the jailbreak probe's score on Z (train OOF scores), split each transfer score into the
persona-explained part Z b and the residual; AUC of score / persona part / residual + R^2 of score on Z; for the full and
span-removed probes; the same with random K-dim span coordinates as the reference (K regressors either way).
usage: python q1_erase.py MODEL CONFIG_JSON Q1_JSON OUTDIR NJOBS [--smoke] [--cell=raw|PersonaJB]"""
import json, os, sys, time, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
M, CFG, SW, OUT, NJ = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4], int(sys.argv[5]); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
TRAIN, TEST, SETS = CFG["train"], CFG["test"], CFG["train"] + CFG["test"]
GRID = [1e-2] if SMOKE else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; B = 1000; ND = 1 if SMOKE else 3
CGRID = [1e-2] if SMOKE else [1e-3, 1e-2, 1e-1, 1.0]
AVG = [s for s in TEST if s != "wjb"]
# ---- loaders: verbatim from q1_sweep.py ----
meta = {s: json.load(open(CFG["meta"].format(set=s))) for s in SETS}; LAYERS, TAIL = meta[SETS[0]]["layers"], meta[SETS[0]]["tail_tokens"]
rows, keep = {}, {}
for s in SETS:
    C = {r["pair_id"]: r for r in map(json.loads, open(CFG["labels"][s]))}; ids = meta[s]["pair_ids"]
    keep[s] = np.array([i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")])
    rows[s] = [C[ids[i]] for i in keep[s]]
y = {s: np.array([int(r["jailbroken"]) for r in rows[s]]) for s in SETS}
Rtr = [r for s in TRAIN for r in rows[s]]; ytr = np.concatenate([y[s] for s in TRAIN])
grp = np.array([f"t{r['jailbreak_idx']}" if r["attack_type"] == "human_jailbreak" else f"b{r['behavior_id']}" for r in Rtr])
FOLDS = list(GroupKFold(5).split(np.zeros(len(ytr)), ytr, grp))
fold = lambda yy, s: max(roc_auc_score(yy, s), 1 - roc_auc_score(yy, s))
TV = np.load(CFG["traits"], mmap_mode="r"); K = TV.shape[0]; D = TV.shape[2]
unit = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)
orth = lambda V: np.linalg.qr(np.asarray(V, np.float64).T)[0]   # [d, k] orthonormal basis of the row span
behs = sorted({r["behavior_id"] for s in SETS for r in rows[s]}); bi = {b: i for i, b in enumerate(behs)}
W = np.random.default_rng(12345).multinomial(len(behs), np.ones(len(behs)) / len(behs), size=B)
BO = {s: np.array([bi[r["behavior_id"]] for r in rows[s]]) for s in AVG}
def boot_avg(test):   # as q1_sweep.boot_avg
    pts, reps = [], []
    for s in AVG:
        yy, sc, bo = y[s], test[s], BO[s]; sg = 1 if roc_auc_score(yy, sc) >= 0.5 else -1; pts.append(roc_auc_score(yy, sg * sc))
        reps.append([roc_auc_score(yy, sg * sc, sample_weight=W[k][bo]) if (W[k][bo] * yy).sum() > 0 and (W[k][bo] * (1 - yy)).sum() > 0 else np.nan for k in range(B)])
    return float(np.mean(pts)), np.nanmean(np.array(reps), 0)
ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
# ---- specificity labels: HarmBench semantic category (HB train rows; transfer rows without copyright) ----
hb_rows = np.arange(len(rows["hb"])) if TRAIN[0] == "hb" else None
assert TRAIN[0] == "hb"
CATS = sorted({r["semantic_category"] for r in rows["hb"]})
cat_tr = np.array([CATS.index(r["semantic_category"]) for r in rows["hb"]]); cat_grp = np.array([r["behavior_id"] for r in rows["hb"]])
CFOLDS = list(GroupKFold(5).split(np.zeros(len(cat_tr)), cat_tr, cat_grp))
cat_te = {s: np.array([CATS.index(r["semantic_category"]) if r["semantic_category"] in CATS else -1 for r in rows[s]]) for s in AVG}


# ---- raw probe (q1_sweep.lr_det with the identity feature map), several inputs in one parallel batch ----
def _fit(Xa, ya, Xb, C):
    sc = StandardScaler().fit(Xa); m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xa), ya)
    return m.decision_function(sc.transform(Xb)).astype(np.float32)


def _final(Xtr, Xte, C):
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
    return {s: m.decision_function(sc.transform(X)).astype(np.float32) for s, X in Xte.items()}


def probes(XS):
    jobs = [(k, f, C) for k in XS for f in range(5) for C in GRID]
    o = Parallel(n_jobs=NJ, backend="loky")(delayed(_fit)(XS[k][0][FOLDS[f][0]], ytr[FOLDS[f][0]], XS[k][0][FOLDS[f][1]], C) for k, f, C in jobs)
    res = {}
    for k in XS:
        vs = {C: np.full(len(ytr), np.nan, np.float32) for C in GRID}; a = {C: [] for C in GRID}
        for (kk, f, C), sc in zip(jobs, o):
            if kk == k: vs[C][FOLDS[f][1]] = sc; a[C].append(fold(ytr[FOLDS[f][1]], sc))
        vauc = {C: float(np.mean(a[C])) for C in GRID}; res[k] = {"C": max(GRID, key=lambda c: vauc[c]), "val_auc": vauc}
        res[k]["oof"] = vs[res[k]["C"]]
    fin = Parallel(n_jobs=min(NJ, len(XS)), backend="loky")(delayed(_final)(XS[k][0], XS[k][1], res[k]["C"]) for k in XS)
    for k, t in zip(XS, fin):
        res[k]["test"] = t; res[k]["avg"], res[k]["reps"] = boot_avg(t)
    return res


# ---- specificity probe: multinomial LR on HB rows -> semantic category, macro OvR AUC ----
def _cfit(Xa, ya, Xb, C):
    sc = StandardScaler().fit(Xa); m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xa), ya)
    return m.predict_proba(sc.transform(Xb)).astype(np.float32)


def macro_auc(lab, P):
    ok = lab >= 0; lab, P = lab[ok], P[ok]
    return float(np.mean([roc_auc_score(lab == c, P[:, c]) for c in range(len(CATS)) if 0 < (lab == c).sum() < len(lab)]))


def cat_probes(XS):
    """XS: {key: (Xhb, {set: Xte})}"""
    jobs = [(k, f, C) for k in XS for f in range(5) for C in CGRID]
    o = Parallel(n_jobs=NJ, backend="loky")(delayed(_cfit)(XS[k][0][CFOLDS[f][0]], cat_tr[CFOLDS[f][0]], XS[k][0][CFOLDS[f][1]], C) for k, f, C in jobs)
    res = {}
    for k in XS:
        a = {C: [] for C in CGRID}
        for (kk, f, C), P in zip(jobs, o):
            if kk == k: a[C].append(macro_auc(cat_tr[CFOLDS[f][1]], P))
        vauc = {C: float(np.mean(a[C])) for C in CGRID}; Cb = max(CGRID, key=lambda c: vauc[c])
        res[k] = {"C": Cb, "val_auc": vauc[Cb]}
    fin = Parallel(n_jobs=min(NJ, len(XS)), backend="loky")(delayed(_cfit)(XS[k][0], cat_tr, XS[k][1][s], res[k]["C"]) for k in XS for s in AVG)
    i = 0
    for k in XS:
        res[k]["test_auc"] = {}
        for s in AVG: res[k]["test_auc"][s] = macro_auc(cat_te[s], fin[i]); i += 1
        res[k]["avg"] = float(np.mean(list(res[k]["test_auc"].values())))
    return res


# ---- GENERAL (non-safety) probes on standard NLP tasks (same chat template, layer and position as the detection acts) ----
#   sst2: SST-2 sentiment (validation, 872); agnews: AG News topic, 4 classes (250 each); cities: Geometry-of-Truth
#   true/false city statements (1496); harder (pjb_gen_acts2.sbatch, gen_inputs2.jsonl): cola (CoLA acceptability), mnli
#   (MNLI 3-way, 400 each), mmlu (MMLU question + correct vs random wrong answer, 750 x 2), negcities (negated cities),
#   claims (common_claim_true_false), arith (3-digit addition, true vs near-miss). Activations from pjb_gen_acts{,2}.sbatch. Edits are fit on the jailbreak training data
#   and only applied here; each probe is retrained per edit (5-fold stratified CV, AUC; macro one-vs-rest for agnews).
from sklearn.model_selection import StratifiedKFold
GA = CFG["ctrl_acts"].replace("ctrl_acts", "gen_acts"); GSETS = ("sst2", "agnews", "cities", "cola", "mnli", "mmlu", "negcities", "claims", "arith")
glab = {(r["set"], r["pair_id"]): r["label"] for f in ("gen_inputs.jsonl", "gen_inputs2.jsonl")
        for r in map(json.loads, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)))}
gmeta = {s: json.load(open(GA.replace("{set}_L{layer}.npy", f"{s}_meta.json"))) for s in GSETS}
for s in GSETS: assert gmeta[s]["tail_tokens"] == TAIL, (s, gmeta[s]["tail_tokens"], TAIL)
GY = {s: np.array([glab[(s, pid)] for pid in gmeta[s]["pair_ids"]]) for s in GSETS}
print("general tasks:", {s: np.bincount(GY[s]).tolist() for s in GSETS}, flush=True)
class _GX(dict):
    def __missing__(self, L):
        self[L] = {s: np.load(GA.format(set=s, layer=L), mmap_mode="r") for s in GSETS}; return self[L]
GX = _GX()


def _gfit(Xa, ya, Xb, yb, C):
    sc = StandardScaler().fit(Xa); m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xa), ya)
    if len(np.unique(ya)) > 2: return roc_auc_score(yb, m.predict_proba(sc.transform(Xb)), multi_class="ovr", average="macro")
    return roc_auc_score(yb, m.decision_function(sc.transform(Xb)))


def gen_probes(XS):
    """XS: {edit: {task: X}} -> {edit: {task: mean 5-fold AUC at the best C (same selection for every edit)}}"""
    jobs = []
    for k, XT in XS.items():
        for t, X in XT.items():
            yy = GY[t]
            for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, yy)):
                for C in CGRID: jobs.append((k, t, C, X[a], yy[a], X[b], yy[b]))
    o = Parallel(n_jobs=NJ, backend="loky")(delayed(_gfit)(j[3], j[4], j[5], j[6], j[2]) for j in jobs)
    acc = {}
    for j, auc in zip(jobs, o): acc.setdefault((j[0], j[1], j[2]), []).append(auc)
    return {k: {t: float(max(np.mean(acc[(k, t, C)]) for C in CGRID)) for t in GSETS} for k in XS}


# ---- multi-output ridge X -> standardized Z, alpha by grouped CV (same folds), via SVD (as q1_reencode.ridge) ----
def r2(Zt, Zp):
    sse = ((Zt - Zp) ** 2).sum(0); sst = ((Zt - Zt.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - sse / sst)), float(1 - sse.sum() / sst.sum())


def ridge(Xtr, Ztr, Xte, Zte):
    zm, zs = Ztr.mean(0), Ztr.std(0); zs[zs == 0] = 1; Zs = (Ztr - zm) / zs
    base = ((Xtr - Xtr.mean(0)) ** 2).sum() / min(Xtr.shape)
    AL = base * np.logspace(-5, 1, 7)
    oof = {a: np.zeros_like(Zs) for a in AL}
    for tr, va in FOLDS:
        xm = Xtr[tr].mean(0); U, s, Vt = np.linalg.svd(Xtr[tr] - xm, full_matrices=False)
        UtZ = U.T @ (Zs[tr] - Zs[tr].mean(0)); XV = (Xtr[va] - xm) @ Vt.T
        for a in AL: oof[a][va] = (XV * (s / (s ** 2 + a))) @ UtZ + Zs[tr].mean(0)
    cv = {a: r2(Zs, oof[a]) for a in AL}; ab = max(AL, key=lambda a: cv[a][0])
    xm = Xtr.mean(0); U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    Bc = Vt.T @ ((s / (s ** 2 + ab))[:, None] * (U.T @ (Zs - Zs.mean(0))))
    te = {t: r2((Zte[t] - zm) / zs, (X - xm) @ Bc + Zs.mean(0))[0] for t, X in Xte.items()}
    return {"alpha_rel": float(ab / base), "cv_r2_mean": cv[ab][0], "cv_r2_vw": r2(Ztr, oof[ab] * zs + zm)[1],
            "transfer_r2_mean": te, "transfer_r2_avg": float(np.mean([te[s] for s in AVG]))}


proj_out = lambda X, U: X - (X @ U) @ U.T
EDITS = ("span", "orth", "leace")


def leace_fit(X, Z):
    """LEACE eraser for target Z (Belrose et al. 2023, Thm 1) with W = (Sigma^{1/2})^+ on the range of Sigma."""
    mu = X.mean(0); Xc = X - mu; S = Xc.T @ Xc / len(X)
    lam, V = np.linalg.eigh(S); ok = lam > lam.max() * 1e-10; lam, V = lam[ok], V[:, ok]
    Sh = (V * np.sqrt(lam)) @ V.T; Wh = (V / np.sqrt(lam)) @ V.T
    Cxz = Xc.T @ (Z - Z.mean(0)) / len(X)
    u, s, _ = np.linalg.svd(Wh @ Cxz, full_matrices=False); U = u[:, s > s.max() * 1e-8]
    A = Sh @ U; Bm = U.T @ Wh                                      # x' = x - A Bm (x - mu)
    return (lambda Xn: Xn - ((Xn - mu) @ Bm.T) @ A.T), int(U.shape[1])


def crosscov_max(X, Z):   # max |corr(x'.e_i , z_j)| proxy: largest singular value of the cross-covariance, relative
    Xc, Zc = X - X.mean(0), Z - Z.mean(0)
    return float(np.linalg.norm(Xc.T @ Zc / len(X), 2) / (np.sqrt((Xc ** 2).sum() / len(X)) * np.sqrt((Zc ** 2).sum() / len(X))))


def run_cell(L, p, tag):
    li = LAYERS.index(L); t0 = time.time()
    X = {s: np.load(CFG["acts"].format(set=s, layer=L), mmap_mode="r")[keep[s], p].astype(np.float64) for s in SETS}
    Xtr0, Xte0 = np.vstack([X[s] for s in TRAIN]), {s: X[s] for s in TEST}; del X
    nhb = len(rows["hb"])
    Qp = orth(unit(np.asarray(TV[:, li], np.float64)))
    Ztr, Zte = Xtr0 @ Qp, {s: Xte0[s] @ Qp for s in TEST}
    mu = Xtr0.mean(0); Xc = Xtr0 - mu; totvar = float((Xc ** 2).sum() / len(Xc))
    ED, FN = {"full": (Xtr0, Xte0, 0)}, {"full": lambda X: X}
    Xs = proj_out(Xtr0, Qp); ED["span"] = (Xs, {s: proj_out(Xte0[s], Qp) for s in TEST}, int(K)); FN["span"] = lambda X: proj_out(X, Qp)
    Cxz = (Xs - Xs.mean(0)).T @ (Ztr - Ztr.mean(0)) / len(Xs); u, sv, _ = np.linalg.svd(Cxz, full_matrices=False); Uo = u[:, sv > sv.max() * 1e-8]
    ED["orth"] = (proj_out(Xs, Uo), {s: proj_out(ED["span"][1][s], Uo) for s in TEST}, int(K + Uo.shape[1])); FN["orth"] = lambda X: proj_out(proj_out(X, Qp), Uo)
    f1, r1 = leace_fit(Xtr0, Ztr); ED["leace"] = (f1(Xtr0), {s: f1(Xte0[s]) for s in TEST}, r1); FN["leace"] = f1
    cen = lambda A: float((((A - Xtr0) - (A - Xtr0).mean(0)) ** 2).sum() / len(A))   # variance removed (centred edit energy)
    info = {}
    for k, (A, _, dims) in ED.items():
        en = cen(A); info[k] = {"dims": dims, "edit_energy": en, "edit_energy_frac": en / totvar,
                                                                 "crosscov_rel": crosscov_max(A, Ztr)}
    # rank-matched UNIFORM random controls: the same number of directions as the edit, isotropic Gaussian (know nothing
    # about the data; seeds 4000+g), projected out of the raw activations
    for g in range(ND):
        rng = np.random.default_rng(4000 + g)
        for k in EDITS:
            n = info[k]["dims"]; U = orth(rng.standard_normal((n, D)))
            key = f"iso_{k}_{g}"; ED[key] = (proj_out(Xtr0, U), {s: proj_out(Xte0[s], U) for s in TEST}, n); FN[key] = (lambda X, U=U: proj_out(X, U))
            A = ED[key][0]; info[key] = {"dims": n, "edit_energy": cen(A), "edit_energy_frac": cen(A) / totvar, "crosscov_rel": crosscov_max(A, Ztr), "matches": k}
    print(f"[{tag}] edits built ({time.time() - t0:.0f}s): " + ", ".join(f"{k} d{v['dims']} E{v['edit_energy_frac']:.3f} cc{v['crosscov_rel']:.1e}" for k, v in info.items()), flush=True)
    jb = probes({k: (v[0], v[1]) for k, v in ED.items()})
    cat = cat_probes({k: (v[0][:nhb], v[1]) for k, v in ED.items()})
    gen = gen_probes({k: {t: FN[k](np.asarray(GX[L][t][:, p], np.float64)) for t in GSETS} for k in ED})
    out = {"layer": L, "position": p, "token": TAIL[p], "k": int(K), "d": int(D), "total_var": totvar, "edits": {}}
    SC = {}
    for k, (A, T, dims) in ED.items():
        rr = ridge(A, Ztr, T, Zte) if k != "full" else {"cv_r2_mean": 1.0, "transfer_r2_avg": 1.0}
        P = jb[k]
        out["edits"][k] = {**info[k], "persona_r2": rr,
                           "jb": {"C": P["C"], "val_auc": P["val_auc"][P["C"]], "test_auc": {s: fold(y[s], P["test"][s]) for s in TEST},
                                  "avg": P["avg"], "avg_ci95": ci(P["reps"]), "delta_vs_full": P["avg"] - jb["full"]["avg"],
                                  "delta_vs_full_ci95": ci(P["reps"] - jb["full"]["reps"])},
                           "category": cat[k], "general": gen[k]}
        for s in TEST: SC[f"{tag}__{k}__test_{s}"] = P["test"][s]
        SC[f"{tag}__{k}__oof"] = P["oof"]; SC[f"boot__{tag}__{k}"] = P["reps"].astype(np.float32)
        print(f"[{tag}] {k:18s} persona R2 cv {rr['cv_r2_mean']:.3f} tr {rr['transfer_r2_avg']:.3f} | jb AUC {P['avg']:.3f} | cat AUC {cat[k]['avg']:.3f} | general " + " ".join(f"{t} {v:.3f}" for t, v in gen[k].items()), flush=True)
    # edit vs its matched controls (paired)
    for k in EDITS:
        cr = np.mean([jb[f"iso_{k}_{g}"]["reps"] for g in range(ND)], 0); ca = float(np.mean([jb[f"iso_{k}_{g}"]["avg"] for g in range(ND)]))
        out["edits"][k]["jb"]["minus_iso_ctl"] = jb[k]["avg"] - ca; out["edits"][k]["jb"]["minus_iso_ctl_ci95"] = ci(jb[k]["reps"] - cr)
    # light score test: how much of the probe's score is explained by the persona coordinates?
    out["score_test"] = {}
    # reference spans for the score test: random K-dim Gaussian spans (seeds 999.. = q1's null span), same K regressors
    spans = {"persona": (Ztr, Zte)}
    for g in range(ND):
        Qr = orth(np.random.default_rng(999 + g).standard_normal((K, D))); spans[f"randspan_{g}"] = (Xtr0 @ Qr, {s: Xte0[s] @ Qr for s in TEST})
    for k in ("full", "span"):
        P = jb[k]; s_tr = P["oof"].astype(np.float64); out["score_test"][k] = {"auc_score": P["avg"], "auc_score_ci95": ci(P["reps"])}
        for sp, (Ztr_, Zte_) in spans.items():
            Zc = np.c_[np.ones(len(Ztr_)), Ztr_]
            beta = np.linalg.solve(Zc.T @ Zc + 1e-6 * np.eye(Zc.shape[1]), Zc.T @ s_tr)
            r2_tr = 1 - ((s_tr - Zc @ beta) ** 2).sum() / ((s_tr - s_tr.mean()) ** 2).sum()
            part = {s: (np.c_[np.ones(len(Zte_[s])), Zte_[s]] @ beta).astype(np.float32) for s in TEST}
            resid = {s: (P["test"][s] - part[s]).astype(np.float32) for s in TEST}
            r2_te = {s: float(1 - ((P["test"][s] - part[s]) ** 2).sum() / ((P["test"][s] - P["test"][s].mean()) ** 2).sum()) for s in TEST}
            pa, preps = boot_avg(part); ra, rreps = boot_avg(resid)
            out["score_test"][k][sp] = {"score_r2_train_oof": float(r2_tr), "score_r2_transfer": r2_te,
                                        "auc_span_part": pa, "auc_span_part_ci95": ci(preps),
                                        "auc_residual": ra, "auc_residual_ci95": ci(rreps), "residual_minus_score_ci95": ci(rreps - P["reps"])}
            for s in TEST: SC[f"{tag}__score_{k}__{sp}_part__test_{s}"] = part[s]; SC[f"{tag}__score_{k}__{sp}_residual__test_{s}"] = resid[s]
            print(f"[{tag}] score test ({k}, {sp}): R2(score~span coords) train {r2_tr:.3f}; AUC score {P['avg']:.3f} span part {pa:.3f} residual {ra:.3f}", flush=True)
    print(f"[{tag}] done ({time.time() - t0:.0f}s)", flush=True)
    return out, SC


cells = {"raw": SW["selected"]["raw__full__reselect"], "PersonaJB": SW["selected"]["PersonaJB__full__reselect"]}
ONLY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cell=")), None)   # run one cell per job
res = {"model": M, "train": TRAIN, "test": TEST, "avg_sets": AVG, "k": int(K), "grid": GRID, "cat_grid": CGRID, "categories": CATS,
       "B": B, "n_draws": ND, "n_train": int(len(ytr)), "n_test": {s: [int(len(y[s])), int(y[s].sum())] for s in TEST}, "cells": {}}
SC = {"labels_train": ytr, "pair_ids_train": np.array([f"{s}:{r['pair_id']}" for s in TRAIN for r in rows[s]])}
for s in TEST: SC[f"labels_{s}"] = y[s]; SC[f"pair_ids_{s}"] = np.array([str(r["pair_id"]) for r in rows[s]]); SC[f"category_{s}"] = cat_te[s]
suf = ("_smoke" if SMOKE else "") + (f"_{ONLY}" if ONLY else ""); fj, fz = f"{OUT}/erase_{M}{suf}.json", f"{OUT}/scores_erase_{M}{suf}.npz"
done = set()
for tag, sel in cells.items():
    L, p = sel["layer"], sel["position"]
    if ONLY and tag != ONLY: continue
    if ONLY and ONLY != "raw" and (L, p) == (cells["raw"]["layer"], cells["raw"]["position"]): print("same cell as raw, nothing to do"); continue
    if (L, p) in done: continue
    done.add((L, p)); key = f"L{L}p{p}"
    o, sc = run_cell(L, p, key); o["selected_by"] = [t for t, s in cells.items() if (s["layer"], s["position"]) == (L, p)]
    res["cells"][key] = o; SC.update(sc)
    json.dump(res, open(fj, "w"), indent=1); np.savez(fz, **SC)
    if SMOKE: break
print("DONE", flush=True)
