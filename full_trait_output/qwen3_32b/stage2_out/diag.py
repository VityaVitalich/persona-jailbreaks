import json, random, numpy as np, torch, collections
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def load(st):
    rows=[json.loads(l) for l in open(f"classified_{st}.jsonl")]; A=torch.load(f"acts_{st}.pt",map_location="cpu")
    rows=[r for r in rows if r["jailbroken"] is not None and r["pair_id"] in A]
    return np.stack([A[r["pair_id"]]["32"].float().numpy() for r in rows]).astype(np.float64), np.array([int(r["jailbroken"]) for r in rows]), rows
Xh,yh,hv=load("hb"); k=[i for i,r in enumerate(hv) if r["attack_type"]=="human_jailbreak"]; Xh,yh,hv=Xh[k],yh[k],[hv[i] for i in k]
Xa,ya,_=load("wjb"); Xp,yp,_=load("pap")
fold=lambda y,s:max(roc_auc_score(y,s),1-roc_auc_score(y,s))
print("HB n",len(yh),"jb",yh.sum(),"| templates with any jb",len({r['jailbreak_idx'] for r,y in zip(hv,yh) if y}),"of",len({r['jailbreak_idx'] for r in hv}),"| requests with any jb",len({r['behavior_id'] for r,y in zip(hv,yh) if y}))
c=collections.Counter(r['jailbreak_idx'] for r,y in zip(hv,yh) if y); print("top templates by jb count",c.most_common(8))
n=np.linalg.norm(Xh,axis=1); mu=Xh.mean(0); print("norm median",np.median(n),"cos w/ mean",np.median(Xh@mu/n/np.linalg.norm(mu)))
top=np.argsort(-np.abs(mu))[:5]; print("share of sq norm in top5 dims",np.median((Xh[:,top]**2).sum(1)/n**2))
T=np.load("../trait_matrix_layer32.npy").astype(np.float64); T/=np.linalg.norm(T,axis=1,keepdims=True)
Q,_=np.linalg.qr(T.T); print("raw act share in trait span",np.median(((Xh@Q)**2).sum(1)/n**2),"| centered",np.median((((Xh-mu)@Q)**2).sum(1)/((Xh-mu)**2).sum(1)))
def split(seed,f=.7):
    rng=random.Random(seed); b=sorted({r["behavior_id"] for r in hv}); t=sorted({r["jailbreak_idx"] for r in hv}); rng.shuffle(b); rng.shuffle(t)
    trb,trt=set(b[:int(len(b)*f)]),set(t[:int(len(t)*f)])
    tr=np.array([i for i,r in enumerate(hv) if r["behavior_id"] in trb and r["jailbreak_idx"] in trt]); te=np.array([i for i,r in enumerate(hv) if r["behavior_id"] not in trb and r["jailbreak_idx"] not in trt]); return tr,te
res=collections.defaultdict(list)
pcaV=np.linalg.svd(Xh-mu,full_matrices=False)[2][:240]
G=np.random.default_rng(0).standard_normal((240,Xh.shape[1])); G/=np.linalg.norm(G,axis=1,keepdims=True)
for s in range(10):
    tr,te=split(s)
    if not(0<yh[te].sum()<len(te)): continue
    Xtr=np.vstack([Xh[tr],Xa]); ytr=np.r_[yh[tr],ya]
    d=Xtr[ytr==1].mean(0)-Xtr[ytr==0].mean(0); res["dmu (HB+WJB)"].append((fold(yh[te],Xh[te]@d),fold(yp,Xp@d)))
    d2=Xh[tr][yh[tr]==1].mean(0)-Xh[tr][yh[tr]==0].mean(0); res["dmu (HB only)"].append((fold(yh[te],Xh[te]@d2),fold(yp,Xp@d2)))
    for nm,V in [("PersonaJB",T),("Gaussian",G),("HB PCA-240",pcaV)]:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=10,max_iter=4000)).fit(Xtr@V.T,ytr); res[nm].append((fold(yh[te],m.decision_function(Xh[te]@V.T)),fold(yp,m.decision_function(Xp@V.T))))
        m=make_pipeline(StandardScaler(),LogisticRegression(C=10,max_iter=4000)).fit(Xh[tr]@V.T,yh[tr]); res[nm+" (HB only)"].append((fold(yh[te],m.decision_function(Xh[te]@V.T)),fold(yp,m.decision_function(Xp@V.T))))
    # train-set fit for PersonaJB
    m=make_pipeline(StandardScaler(),LogisticRegression(C=10,max_iter=4000)).fit(Xtr@T.T,ytr); res["PersonaJB train-fit"].append((fold(yh[tr],m.decision_function(Xh[tr]@T.T)),fold(ya,m.decision_function(Xa@T.T))))
    # template-only baseline: per-template train jb rate
    rate=collections.defaultdict(float); cnt=collections.Counter()
    for i in tr: rate[hv[i]['jailbreak_idx']]+=yh[i]; cnt[hv[i]['jailbreak_idx']]+=1
    req=collections.defaultdict(float); rc=collections.Counter()
    for i in tr: req[hv[i]['behavior_id']]+=yh[i]; rc[hv[i]['behavior_id']]+=1
print({k:f"{v:.3f}" for k,v in [("prompt length AUC (HB test, all)",fold(yh,np.array([len(r['response_text']) for r in hv])))]})
for k,v in res.items(): v=np.array(v); print(f"{k:28s} HB {v[:,0].mean():.3f}  PAP {v[:,1].mean():.3f}   (n splits {len(v)})")
# WJB in-distribution CV
from sklearn.model_selection import cross_val_score
for nm,V in [("PersonaJB",T),("Gaussian",G)]:
    print("WJB 5-fold AUC",nm,cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(C=10,max_iter=4000)),Xa@V.T,ya,cv=5,scoring="roc_auc").mean().round(3))
