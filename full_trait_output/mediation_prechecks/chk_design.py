import json, numpy as np, torch, collections
rows=[json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
A=torch.load('harmbench_activations/activations.pt', map_location='cpu', weights_only=False)
H={r['pair_id']:A[r['pair_id']]['16'].float().numpy().astype(np.float64) for r in rows}
key={(r['behavior_id'],r['jailbreak_idx']):r for r in rows}
behs=sorted({r['behavior_id'] for r in rows})
T=np.load('trait_matrix_layer16.npy').astype(np.float64)
Q=np.linalg.svd(T,full_matrices=False)[2].T
rng=np.random.default_rng(0)
Qr=[np.linalg.qr(rng.standard_normal((4096,229)))[0] for _ in range(5)]
X=np.stack(list(H.values())); Qp=np.linalg.svd(X-X.mean(0),full_matrices=False)[2]
def share(D,B): return np.mean(np.sum((D@B)**2,1)/np.sum(D**2,1))
weak=[18,73,30,29,71]; strong=[11,3,32,77]
def pairs(xs,ys):
    D=[];y=[]
    for b in behs:
        for i in xs:
            for j in ys:
                D.append(H[key[(b,j)]['pair_id']]-H[key[(b,i)]['pair_id']]); y.append((key[(b,i)]['jailbroken'],key[(b,j)]['jailbroken']))
    return np.array(D), y
for name,(xs,ys) in {'DR->all20':([-1],[k for k in {r['jailbreak_idx'] for r in rows} if k>=0]),'weak->strong':(weak,strong),'weak->weak':(weak,weak),'strong->strong':(strong,strong)}.items():
    D,y=pairs(xs,ys); D=D[np.linalg.norm(D,axis=1)>0]
    pn=np.linalg.norm(D@Q,axis=1)
    print(f"{name}: n={len(D)} shareT={share(D,Q):.3f} rand={np.mean([share(D,q) for q in Qr]):.3f} pca229={share(D,Qp[:229].T):.3f} pca5={share(D,Qp[:5].T):.3f} pca2={share(D,Qp[:2].T):.3f} med|D|={np.median(np.linalg.norm(D,axis=1)):.2f} med|PD|={np.median(pn):.2f}")
# norms of h at -1
print('median |h| weak', np.median([np.linalg.norm(H[key[(b,i)]['pair_id']]) for b in behs for i in weak]))
# refusal direction layers
md=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/mean_diffs.pt',map_location='cpu').double().numpy()
print('mean_diffs shape', md.shape)
for l in [16,17]:
    r=md[-1,l]/np.linalg.norm(md[-1,l]); print('share md[-1,%d]'%l, np.sum((Q.T@r)**2))
r=md[-1,17]/np.linalg.norm(md[-1,17])
D,y=pairs(weak,strong)
print('mean D.r', np.mean(D@r), 'from PD', np.mean((D@Q)@(Q.T@r)))
# energy-matched feasibility: rank-229 subspaces drawn from N(0, Sigma_act)
Xc=X-X.mean(0); D,_=pairs(weak,strong)
U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
sh=[]
for s in range(10):
    G=(rng.standard_normal((229,len(S)))*S)@Vt   # rows ~ N(0,Sigma)
    q=np.linalg.qr(G.T)[0]; sh.append(share(D,q))
print('N(0,Sigma) rank229 shares', np.round(sh,3))
# per-row distribution of share for T on weak->strong
s=np.sum((D@Q)**2,1)/np.sum(D**2,1); print('T share per-pair quantiles', np.round(np.quantile(s,[.1,.5,.9]),3))
for K in [400,700,1000,1500]:
    sh=[]
    for s in range(3):
        R=np.linalg.qr(rng.standard_normal((K,229)))[0]; q=Vt[:K].T@R; sh.append(share(D,q))
    print('rot within top-%d PCs'%K, np.round(sh,3))
for g in [0.0,0.25,0.5]:
    G=(rng.standard_normal((229,len(S)))*S**g)@Vt; q=np.linalg.qr(G.T)[0]; print('Sigma^%.2f'%g, round(share(D,q),3))
