import json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
rng=np.random.default_rng(2)
T=np.load('trait_matrix_layer16.npy').astype(np.float64); Q=np.linalg.svd(T,full_matrices=False)[2].T
A=torch.load('harmbench_activations/activations.pt',weights_only=False)
R=[json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
H={r['pair_id']:A[r['pair_id']]['16'].float().numpy().astype(np.float64) for r in R}
cell={(r['jailbreak_idx'],r['behavior_id']):r['pair_id'] for r in R}
behs=sorted({r['behavior_id'] for r in R}); W=[18,73,30,29,71]; S=[11,3,32,77]
h0=np.stack([H[cell[(t0,b)]] for b in behs for t0 in W for t1 in S]); h1=np.stack([H[cell[(t1,b)]] for b in behs for t0 in W for t1 in S])
D=h1-h0; hs=h0+D@Q@Q.T
print('||h*||/||h0|| quantiles 1,10,50,90,99%',np.round(np.quantile(np.linalg.norm(hs,axis=1)/np.linalg.norm(h0,axis=1),[.01,.1,.5,.9,.99]),3))
X=np.stack(list(H.values())); mu=X.mean(0); U,s,Vt=np.linalg.svd(X-mu,full_matrices=False)
k=500; lam=s[:k]**2/len(X); 
def maha(Z): z=(Z-mu)@Vt[:k].T; res=np.sum((Z-mu)**2,1)-np.sum(z**2,1); return np.sqrt(np.sum(z**2/lam,1)), res
for name,Z in [('h0',h0),('h1',h1),('h* T',hs),('h* haar',h0+D@(lambda B:B@B.T)(np.linalg.qr(rng.standard_normal((4096,229)))[0])),('h0+P(mu1-mu0) mean-shift',h0+((h1.mean(0)-h0.mean(0))@Q@Q.T))]:
    m,res=maha(Z); print(f'{name:28s} maha(top{k}) median {np.median(m):.1f}  p90 {np.quantile(m,.9):.1f}  resid-norm^2 median {np.median(res):.2f}')
