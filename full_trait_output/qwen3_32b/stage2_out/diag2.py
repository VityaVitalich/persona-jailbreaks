exec(open("diag.py").read().split("res=collections")[0].replace('print(','(lambda *a,**k:None)('))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
mk=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=10,max_iter=4000))
G=np.random.default_rng(0).standard_normal((240,Xh.shape[1])); G/=np.linalg.norm(G,axis=1,keepdims=True)
tpl=np.array([r['jailbreak_idx'] for r in hv]); beh=np.array([r['behavior_id'] for r in hv])
def within_req(y,s):
    a=[]
    for b in set(beh):
        m=beh==b
        if 0<y[m].sum()<m.sum(): a.append(roc_auc_score(y[m],s[m]))
    return np.mean(a), len(a)
for nm,V in [("PersonaJB",T),("Gaussian",G)]:
    Z=Xh@V.T; aucs=[];wr=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Z,yh):
        s=mk().fit(np.vstack([Z[tr],Xa@V.T]),np.r_[yh[tr],ya]).decision_function(Z); aucs.append(roc_auc_score(yh[te],s[te]))
    print(f"{nm:10s} random prompt split HB AUC {np.mean(aucs):.3f}")
    # leave-one-template-out: score each held-out template's prompts, then pooled AUC and within-request AUC
    s=np.zeros(len(yh))
    for t in set(tpl):
        tr=tpl!=t; s[~tr]=mk().fit(np.vstack([Z[tr],Xa@V.T]),np.r_[yh[tr],ya]).decision_function(Z[~tr])
    w,nb=within_req(yh,s); print(f"{nm:10s} leave-one-template-out pooled AUC {fold(yh,s):.3f}  within-request AUC {w:.3f} over {nb} requests")
    per=[(t,yh[tpl==t].mean(),s[tpl==t].mean()) for t in sorted(set(tpl))]
    from scipy.stats import spearmanr
    print("   template jb-rate vs mean held-out score spearman", round(spearmanr([p[1] for p in per],[p[2] for p in per])[0],3))
# template identity oracle within request
print("template-rate oracle within-request AUC", within_req(yh, np.array([yh[tpl==t].mean() for t in tpl]))[0].round(3))
