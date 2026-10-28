import json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
rng=np.random.default_rng(1)
T=np.load('trait_matrix_layer16.npy').astype(np.float64); Q=np.linalg.svd(T,full_matrices=False)[2].T
A=torch.load('harmbench_activations/activations.pt',weights_only=False)
H=np.stack([v['16'].float().numpy() for v in A.values()]).astype(np.float64)
D=np.load('/private/tmp/claude-503/-Users-viktor/7ed7bcdc-315c-478e-8a5c-8b65132aee39/scratchpad/_D.npy')
md=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/mean_diffs.pt',weights_only=False,map_location='cpu').numpy().astype(np.float64)
r=md[4,17]; r/=np.linalg.norm(r)
sh=lambda X,B: np.mean(np.sum((X@B)**2,1)/np.sum(X**2,1))
print('T share',sh(D,Q),' r-share',np.sum((r@Q)**2))
Hc=H-H.mean(0); U,s,Vt=np.linalg.svd(Hc,full_matrices=False)  # rank<=3338
print('PCs available',len(s))
# N(0,Sigma) sampled subspaces
for g in [1.0,0.5,0.25]:
  vals=[];rs=[]
  for _ in range(5):
    Z=rng.standard_normal((229,len(s)))*(s**g); B=np.linalg.qr((Z@Vt).T)[0]
    vals.append(sh(D,B)); rs.append(np.sum((r@B)**2))
  print(f'N(0,Sigma^{g}) rank229: share {np.mean(vals):.3f}+-{np.std(vals):.3f}  r-share {np.mean(rs):.3f}')
# rotations within top-K PCs
for K in [400,600,800,1200]:
  vals=[];rs=[]
  for _ in range(10):
    Rk=np.linalg.qr(rng.standard_normal((K,229)))[0]; B=Vt[:K].T@Rk
    vals.append(sh(D,B)); rs.append(np.sum((r@B)**2))
  print(f'rot top-{K}: share {np.mean(vals):.3f}+-{np.std(vals):.3f} (min {np.min(vals):.3f} max {np.max(vals):.3f}) r-share {np.mean(rs):.3f}+-{np.std(rs):.3f}')
# per-pair share distribution T vs one matched
print('T per-pair share quantiles',np.quantile(np.sum((D@Q)**2,1)/np.sum(D**2,1),[.1,.5,.9]))
