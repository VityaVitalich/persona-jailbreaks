import json, numpy as np, torch, warnings, itertools
warnings.filterwarnings('ignore')
T=np.load('trait_matrix_layer16.npy').astype(np.float64); Q=np.linalg.svd(T,full_matrices=False)[2].T
A=torch.load('harmbench_activations/activations.pt',weights_only=False)
R=[json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
H={}; Y={}
for r in R:
    H[(r['jailbreak_idx'],r['behavior_id'])]=A[r['pair_id']]['16'].float().numpy().astype(np.float64)
    Y[(r['jailbreak_idx'],r['behavior_id'])]=int(bool(r['jailbroken']))
behs=sorted({r['behavior_id'] for r in R})
W=[18,73,30,29,71]; S=[11,3,32,77]; WP=[96,79,14]
def stats(name,pairs):
    D=np.stack([H[b]-H[a] for a,b in pairs]); P=D@Q
    sh=np.sum(P**2,1)/np.sum(D**2,1); n=np.linalg.norm(P,axis=1)
    print(f'{name:28s} n={len(D):5d} share={sh.mean():.3f} median||PD||={np.median(n):.2f} median||D||={np.median(np.linalg.norm(D,axis=1)):.2f}')
    return D
mk=lambda G0,G1:[((t0,b),(t1,b)) for b in behs for t0 in G0 for t1 in G1 if t0!=t1]
D_ws=stats('weak->strong same beh',mk(W,S))
stats('DR->strong',mk([-1],S))
stats('weak->weak placebo',mk(W,W))
stats('strong->strong placebo',mk(S,S))
stats('weakpersona->strong',mk(WP,S))
stats('DR->weak',mk([-1],W))
# centred variance share
Hall=np.stack(list(H.values())); Hc=Hall-Hall.mean(0)
print('share of h-mu', np.mean(np.sum((Hc@Q)**2,1)/np.sum(Hc**2,1)))
# ASR
for t in W+S+WP+[-1]: print(t, np.mean([Y[(t,b)] for b in behs]).round(3), end='; ')
print()
# refusal coordinate share
md=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/mean_diffs.pt',weights_only=False,map_location='cpu').numpy()
r=md[4,17]; r=r/np.linalg.norm(r)
pr=D_ws@r; prP=(D_ws@Q)@(Q.T@r)
print('weak->strong: mean D.r', pr.mean().round(3),' from P part', prP.mean().round(3), 'share', (prP.mean()/pr.mean()).round(3))
# T minus r: how much of r-projection-like refusal info remains? fraction of in-span projection on dmu
# design effect: TE pairs clustered by behaviour
d=np.array([[Y[(t1,b)]-Y[(t0,b)] for t0 in W for t1 in S] for b in behs]) # 159 x 20
m=d.mean(); iid_se=d.std()/np.sqrt(d.size); cl_se=d.mean(1).std(ddof=1)/np.sqrt(len(behs))
print('TE',m.round(3),'iid SE',iid_se.round(4),'behaviour-cluster SE',cl_se.round(4),'DE',(cl_se/iid_se)**2)
y0=np.array([[Y[(t0,b)] for t0 in W] for b in behs]); print('y0 mean',y0.mean().round(3))
# ICC of y0 within behaviour
print('--- split relabel sim')
temps=sorted({r['jailbreak_idx'] for r in R if r['jailbreak_idx']>=0})
rng=np.random.default_rng(0); same=0; ex=[]
for s in range(1000):
    A_=set(rng.choice(behs,80,replace=False))
    asr={t:np.mean([Y[(t,b)] for b in A_]) for t in temps}
    w=sorted([t for t in temps if asr[t]<=0.10]); st=sorted([t for t in temps if asr[t]>=0.80])
    if w==sorted(W) and st==sorted(S): same+=1
    else: ex.append((tuple(w),tuple(st)))
print('identical sets',same/1000)
from collections import Counter; print(Counter(ex).most_common(4))
# energy-matched: gaussian-with-cov rank-229 subspaces share on weak->strong D
Hc=Hall-Hall.mean(0); C=np.cov(Hc.T)
L=np.linalg.cholesky(C+1e-6*np.eye(4096))
shs=[]
for i in range(5):
    G=L@rng.standard_normal((4096,229)); Qg=np.linalg.qr(G)[0]
    P=D_ws@Qg; shs.append(np.mean(np.sum(P**2,1)/np.sum(D_ws**2,1)))
print('N(0,Sigma) rank-229 share', np.round(shs,3))
