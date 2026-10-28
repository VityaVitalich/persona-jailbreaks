"""Figure inputs for the refusal / jailbreak direction rows.

Jailbreak dir (dmu): refit per seed on the matched 70/30 train pool (Gemma +WJB)
  -> seed CI, 1.96*sd/sqrt(n) over the per-seed 5-family macro averages.
Refusal dir (pos -1): a fixed vector estimated on Arditi's data, so it has no
  seed variance -> 1000-resample test-set bootstrap, the convention the other
  deterministic rows in the figure use.
"""
import json, statistics, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO/"full_trait_tools"))
from run_all_traits_sweep_v2 import build_activation_matrix, get_pool_split, load_jsonl, split_by_pool
FT = REPO/"full_trait_output"; FAMS = ["GCG","PAIR","PAP","GPTFuzz","PEZ"]
flip = lambda a: max(a, 1-a)
auc = lambda y,s: float(roc_auc_score(y,s))
unit = lambda v: np.asarray(v,dtype=np.float64).ravel()/np.linalg.norm(np.asarray(v,dtype=np.float64).ravel())
CFG = {"llama": dict(layer=16, hb="harmbench_activations", ff="responses.jsonl",
                     fam={f: f"{f.lower()}_activations" for f in FAMS}, aug=None),
       "gemma": dict(layer=30, hb="harmbench_activations_gemma", ff="classified_responses.jsonl",
                     fam={f: f"{f.lower()}_activations_gemma_hb" for f in FAMS},
                     aug="wildjailbreak_activations_gemma")}
def load(d,f,L,filt=False):
    rows=load_jsonl(FT/d/f)
    if filt: rows=[r for r in rows if r.get("attack_type")=="human_jailbreak"]
    x,y,v=build_activation_matrix(rows, torch.load(FT/d/"activations.pt",map_location="cpu",weights_only=False), L)
    return x.astype(np.float64), y, v
out={}
for tag,c in CFG.items():
    L=c["layer"]
    x_hb,y_hb,valid=load(c["hb"],"classified_responses.jsonl",L,True)
    fam={f: load(c["fam"][f], c["ff"], L)[:2] for f in FAMS}
    aug=load(c["aug"],"classified_responses.jsonl",L)[:2] if c["aug"] else None
    # refusal pos -1
    if tag=="llama":
        r=unit(torch.load(Path.home()/"refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt",
                          map_location="cpu",weights_only=False)["direction"].numpy())
    else:
        B=FT/"gemma_refusal_dir"
        a=torch.load(B/"out/means_harmful.pt",map_location="cpu",weights_only=False)[31,4]
        b=torch.load(B/"out/means_harmless.pt",map_location="cpu",weights_only=False)[31,4]
        r=unit((a-b).float().numpy())
    d=-r
    per={f: flip(auc(fam[f][1], fam[f][0]@d)) for f in FAMS}
    pt=statistics.fmean(per.values()); hb=flip(auc(y_hb, x_hb@d))
    rng=np.random.RandomState(0); sc={f:(fam[f][0]@d, fam[f][1]) for f in FAMS}; boot=[]
    for _ in range(1000):
        vals=[]
        for f in FAMS:
            s,y=sc[f]
            while True:
                idx=rng.randint(0,len(y),len(y))
                if len(np.unique(y[idx]))==2: break
            vals.append(flip(auc(y[idx], s[idx])))
        boot.append(statistics.fmean(vals))
    lo,hi=np.percentile(boot,[2.5,97.5])
    out[f"{tag}_refusal"]={"transfer":pt,"ci":float((hi-lo)/2),"hb":hb,"per_family":per}
    # jailbreak dmu
    ps={k:[] for k in ["HarmBench"]+FAMS}
    for seed in range(50):
        tb,tt,eb,et=get_pool_split(valid,0.7,seed); tr,te=split_by_pool(valid,tb,tt,eb,et)
        if len(np.unique(y_hb[tr]))<2 or len(np.unique(y_hb[te]))<2: continue
        xt,yt=(x_hb[tr],y_hb[tr]) if aug is None else (np.vstack([x_hb[tr],aug[0]]),np.concatenate([y_hb[tr],aug[1]]))
        dd=unit(xt[yt==1].mean(0)-xt[yt==0].mean(0))
        ps["HarmBench"].append(auc(y_hb[te], x_hb[te]@dd))
        for f in FAMS: ps[f].append(auc(fam[f][1], fam[f][0]@dd))
    n=len(ps["HarmBench"]); xf=[statistics.fmean(flip(ps[f][i]) for f in FAMS) for i in range(n)]
    out[f"{tag}_jailbreak"]={"transfer":statistics.fmean(xf),
        "ci":1.96*statistics.stdev(xf)/n**0.5, "hb":flip(statistics.fmean(ps["HarmBench"])),
        "per_family":{f: flip(statistics.fmean(ps[f])) for f in FAMS}, "n_seeds":n}
for k,v in out.items():
    print(f"{k:18s} transfer {v['transfer']:.4f} +-{v['ci']:.4f}  HarmBench {v['hb']:.4f}  "
          + " ".join(f"{f} {v['per_family'][f]:.3f}" for f in FAMS))
(REPO / "full_trait_output/paper_figs_inputs/dirs_fig_inputs.json").write_text(json.dumps(out,indent=2))
