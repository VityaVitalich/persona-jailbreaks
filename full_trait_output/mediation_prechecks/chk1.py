import json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
rng=np.random.default_rng(0)
T=np.load('trait_matrix_layer16.npy').astype(np.float64); Q=np.linalg.svd(T,full_matrices=False)[2].T
A=torch.load('harmbench_activations/activations.pt',weights_only=False)
R=[json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
H={r['pair_id']:A[r['pair_id']]['16'].float().numpy().astype(np.float64) for r in R}
W=[18,73,30,29,71]; S=[11,3,32,77]
cell={(r['jailbreak_idx'],r['behavior_id']):r for r in R}
behs=sorted({r['behavior_id'] for r in R})
pairs=[]
for b in behs:
  for t0 in W:
    for t1 in S:
      r0=cell[(t0,b)]; r1=cell[(t1,b)]
      pairs.append((b,t0,t1,H[r0['pair_id']],H[r1['pair_id']],int(r0['jailbroken']),int(r1['jailbroken'])))
h0=np.stack([p[3] for p in pairs]); h1=np.stack([p[4] for p in pairs]); D=h1-h0
y0=np.array([p[5] for p in pairs]); y1=np.array([p[6] for p in pairs]); bb=np.array([p[0] for p in pairs])
sh=lambda X,B: np.mean(np.sum((X@B)**2,1)/np.sum(X**2,1))
print('n pairs',len(pairs),'y0',y0.mean(),'y1',y1.mean())
print('weak->strong share in T', sh(D,Q), ' DR->tmpl was 0.375')
PD=D@Q@Q.T
print('median ||D||',np.median(np.linalg.norm(D,axis=1)),' median ||P D||',np.median(np.linalg.norm(PD,axis=1)))
hs=h0+PD
n0=np.linalg.norm(h0,axis=1); n1=np.linalg.norm(h1,axis=1); ns=np.linalg.norm(hs,axis=1)
print('median ||h0||',np.median(n0),'||h1||',np.median(n1),'||h*||',np.median(ns),' median ||h*||/||h0||',np.median(ns/n0), ' frac h* norm outside [min(h0,h1),max]:',np.mean((ns<np.minimum(n0,n1))|(ns>np.maximum(n0,n1))))
# random haar patches norm
Qr=np.linalg.qr(rng.standard_normal((4096,229)))[0]
hr=h0+D@Qr@Qr.T; print('haar ||h*||/||h0|| median',np.median(np.linalg.norm(hr,axis=1)/n0))
# mean-shift fraction
mu0={t:np.mean([H[cell[(t,b)]['pair_id']] for b in behs],0) for t in W}; mu1={t:np.mean([H[cell[(t,b)]['pair_id']] for b in behs],0) for t in S}
M0=np.mean(list(mu0.values()),0); M1=np.mean(list(mu1.values()),0)
mshift=(M1-M0)@Q@Q.T
print('||P(mu1-mu0)||^2 / mean||P D||^2 =', mshift@mshift/np.mean(np.sum(PD**2,1)))
# template-pair means
tp=np.stack([ (mu1[p[2]]-mu0[p[1]])@Q@Q.T for p in pairs])
print('template-pair-mean P shift explains', 1-np.mean(np.sum((PD-tp)**2,1))/np.mean(np.sum(PD**2,1)), 'of mean ||PD||^2 (rest = behaviour-specific)')
# also within-template variability of P h1 across behaviours
# refusal direction (layer matched)
md=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/mean_diffs.pt',weights_only=False,map_location='cpu').numpy().astype(np.float64)
r=md[4,17]; r/=np.linalg.norm(r)
print('share r17 in T',np.sum((r@Q)**2))
print('mean D.r',np.mean(D@r),' mean PD.r',np.mean(PD@r))
# naive T-minus-r construction rank and leakage
Qn=(np.eye(4096)-np.outer(r,r))@Q
s=np.linalg.svd(Qn,compute_uv=False); print('naive (I-rr^T)Q rank',np.sum(s>1e-8), ' min sv',s.min())
Qn_o=np.linalg.svd(Qn,full_matrices=False)[0]
print('naive span: frac outside span(T)', 1-np.sum((Qn_o.T@Q)**2)/229)
u=Q@(Q.T@r); u/=np.linalg.norm(u)
Qc=np.linalg.svd((np.eye(4096)-np.outer(u,u))@Q,full_matrices=False)
print('correct construction rank', np.sum(Qc[1]>1e-8))
# per-position refusal directions overlap with T
for j in range(5):
  v=md[j,17]/np.linalg.norm(md[j,17]); print('pos',j-5,'share in T',round(float(np.sum((v@Q)**2)),3),'cos with pos-1 r',round(float(v@r),3))
# behaviour-cluster design effect for TE differences and y1
def deff(d,cl):
  n=len(d); dm=d-d.mean(); cs={}
  for x,c in zip(dm,cl): cs[c]=cs.get(c,0)+x
  v_cl=sum(v*v for v in cs.values())/n**2; v_iid=np.var(d)/n; return v_cl/v_iid
print('deff behaviour TE (y1-y0):',deff(y1-y0,bb),' y1:',deff(y1,bb),' y0:',deff(y0,bb))
x0id=np.array([f'{p[1]}_{p[0]}' for p in pairs]); x1id=np.array([f'{p[2]}_{p[0]}' for p in pairs])
print('deff x0-prompt TE',deff(y1-y0,x0id),' x1-prompt',deff(y1-y0,x1id))
np.save('/private/tmp/claude-503/-Users-viktor/7ed7bcdc-315c-478e-8a5c-8b65132aee39/scratchpad/_D.npy',D)
