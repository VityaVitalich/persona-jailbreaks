exec(open("diag.py").read().split("res=collections")[0].replace('print(','(lambda *a,**k:None)('))
import warnings; warnings.filterwarnings("ignore")
tpl=np.array([r['jailbreak_idx'] for r in hv])
res=collections.defaultdict(list)
for s in range(10):
    tr,te=split(s)
    if not(0<yh[te].sum()<len(te)): continue
    for C in (0.001,0.01,0.1,1.0):
        for wj in (True,False):
            Xtr=np.vstack([Xh[tr],Xa]) if wj else Xh[tr]; ytr=np.r_[yh[tr],ya] if wj else yh[tr]
            m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000)).fit(Xtr,ytr)
            res[(C,wj)].append((fold(yh[te],m.decision_function(Xh[te])),fold(yp,m.decision_function(Xp)),
                                roc_auc_score(yh[tr],m.decision_function(Xh[tr])), yh[te].sum(), len(set(tpl[te][yh[te]==1]))))
    print("split",s,"done",flush=True)
for (C,wj),v in sorted(res.items()):
    v=np.array(v,float); print(f"full-act LR C={C:<6} +WJB={wj!s:5} HB test {v[:,0].mean():.3f} (min {v[:,0].min():.2f} max {v[:,0].max():.2f})  PAP {v[:,1].mean():.3f}  train {v[:,2].mean():.3f}")
print("per split test positives / jb templates in test:",[(int(a),int(b)) for a,b in np.array(res[(0.01,True)])[:,3:5]])
