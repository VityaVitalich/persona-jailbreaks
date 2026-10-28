#!/usr/bin/env python3
"""Gemma-4-31B detection-head latencies, mirroring time_detection_heads.py /
time_heads_extra.py exactly (same bench protocol, same estimator classes), so the
numbers sit alongside the Llama tab:timing values.

Heads timed on REAL Gemma activations:
  trait (PersonaJB): layer-30 (5376) @ trait matrix (239) -> standardize -> dot
  raw (Activation Probe): standardize(5376) -> dot
  jbshield: MeanDirectionDetector on layer-30
  pca3: PCA(3) on layer-30 -> scaler+logreg
  jlt_svm / jlt_rf: rtv 3 layers x 5 tokens (80640) -> PCA(128) -> SVC / RF
  mcd: rtv matrix (16128) -> Std -> PCA(128) -> Mahalanobis contrastive
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch

import os
BASE = next(c for c in ("/dlabscratch1","/mnt/dlabscratch1","/mnt/dlab/scratch/dlabscratch1") if Path(c, "bazina").is_dir())
REPO = Path(f"{BASE}/bazina/assistant-axis-llama3.1-8B")
sys.path.insert(0, str(REPO / "full_trait_tools"))
os.chdir(REPO)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from run_all_traits_sweep_v2 import (load_jsonl, build_activation_matrix,
                                     load_trait_matrix, project_all_traits)
from run_baselines_jbshield_fjd import MeanDirectionDetector
import run_baseline_jlt_hidden_tensor as JLT
import run_baseline_contrastive_geometry as MCDmod

LAYER = 30
SEED = 0
CLS = "full_trait_output/harmbench_activations_gemma/classified_responses.jsonl"
ACT = "full_trait_output/harmbench_activations_gemma/activations.pt"
TRAIT_DIR = "full_trait_output/gemma_vectors/filter_matched_pairs_ge_50_count_ge_10_total"
RTV_ROOT = Path("full_trait_output/rtv_activations_gemma")


def bench1(fn, X, n_warm=300, n_rep=10000):
    n = len(X)
    for i in range(n_warm):
        fn(X[i % n])
    t0 = time.perf_counter()
    for i in range(n_rep):
        fn(X[i % n])
    return (time.perf_counter() - t0) / n_rep * 1e6  # us/sample


rows = load_jsonl(Path(CLS))
acts = torch.load(ACT, map_location="cpu", weights_only=False)
X30, y30, _ = build_activation_matrix(rows, acts, LAYER)
X30 = np.ascontiguousarray(X30, dtype=np.float32)
tm = np.load(f"{os.environ['OUT_DIR']}/../trait_matrix_layer30.npy").astype(np.float32)
tm = tm / np.maximum(np.linalg.norm(tm, axis=1, keepdims=True), 1e-12)
tmT = np.ascontiguousarray(tm.T, dtype=np.float32)
X_trait = project_all_traits(X30, tm)
print(f"layer30: N={len(X30)} dim={X30.shape[1]} traits={tm.shape[0]}", flush=True)


def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, penalty="l2",
                                                solver="lbfgs", max_iter=4000,
                                                random_state=0))])


raw_m = mk().fit(X30, y30)
trait_m = mk().fit(X_trait, y30)


def params(m):
    sc, clf = m.named_steps["sc"], m.named_steps["clf"]
    return (sc.mean_.astype(np.float32), sc.scale_.astype(np.float32),
            clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0]))


rmu, rsd, rw, rb = params(raw_m)
tmu, tsd, tw, tb = params(trait_m)
raw_np = lambda x: float(((x - rmu) / rsd) @ rw + rb)
trait_np = lambda x: float((((x @ tmT) - tmu) / tsd) @ tw + tb)

jb = MeanDirectionDetector().fit(X30, y30)
pca3 = PCA(n_components=3, random_state=SEED).fit(X30)
pca3_clf = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=4000,
                                                random_state=SEED))])
pca3_clf.fit(pca3.transform(X30), y30)
jb_fn = lambda x: float(jb.score(x.reshape(1, -1))[0])
pca3_fn = lambda x: float(pca3_clf.decision_function(pca3.transform(x.reshape(1, -1)))[0])

X_jlt, jlt_rows = JLT.load_dataset_tensor(RTV_ROOT, "harmbench")
y_jlt = JLT.labels_for_rows(jlt_rows)
X_jlt = np.ascontiguousarray(X_jlt, dtype=np.float32)
X_mcd, mcd_rows = MCDmod.load_dataset_matrix(RTV_ROOT, "harmbench")
y_mcd = MCDmod.labels_for_rows(mcd_rows)
X_mcd = np.ascontiguousarray(X_mcd, dtype=np.float32)
print(f"jlt: N={len(X_jlt)} dim={X_jlt.shape[1]} | mcd: N={len(X_mcd)} dim={X_mcd.shape[1]}",
      flush=True)

n_comp_j = min(JLT.N_COMPONENTS, X_jlt.shape[0] - 1, X_jlt.shape[1])
pca_j = PCA(n_components=n_comp_j, svd_solver="randomized", random_state=SEED)
Xjp = pca_j.fit_transform(X_jlt)
models = JLT.make_models(SEED, fast=False)
svm = models["pca_svm_rbf"]; svm.fit(Xjp, y_jlt)
rf = models["pca_random_forest"]; rf.fit(Xjp, y_jlt)
rf.n_jobs = 1
n_comp_m = min(MCDmod.N_COMPONENTS, X_mcd.shape[0] - 1, X_mcd.shape[1])
sc_m = StandardScaler().fit(X_mcd)
pca_m = PCA(n_components=n_comp_m, svd_solver="randomized", random_state=SEED)
Xmp = pca_m.fit_transform(sc_m.transform(X_mcd))
mcd = MCDmod.MCD().fit(Xmp, y_mcd)
svm_fn = lambda x: float(svm.decision_function(pca_j.transform(x.reshape(1, -1)))[0])
rf_fn = lambda x: float(rf.predict_proba(pca_j.transform(x.reshape(1, -1)))[0, 1])
mcd_fn = lambda x: float(mcd.score(pca_m.transform(sc_m.transform(x.reshape(1, -1))))[0])

print("\n=== Gemma heads, per-request batch=1 (us/sample -> ms) ===", flush=True)
res = {}
for name, fn, X, warm, rep in [
        ("trait", trait_np, X30, 500, 20000),
        ("raw", raw_np, X30, 500, 20000),
        ("jbshield", jb_fn, X30, 300, 10000),
        ("pca3", pca3_fn, X30, 300, 10000),
        ("jlt_svm", svm_fn, X_jlt, 100, 2000),
        ("jlt_rf", rf_fn, X_jlt, 50, 500),
        ("mcd", mcd_fn, X_mcd, 100, 2000)]:
    us = bench1(fn, X, n_warm=warm, n_rep=rep)
    res[name] = us / 1000.0
    print(f"  {name:10s} {us:10.3f} us = {us/1000:.4f} ms", flush=True)

import json
out = Path(os.environ["OUT_DIR"]) / "head_timings.json"
out.write_text(json.dumps(res, indent=1))
print("saved", out)
