import json, numpy as np, torch
rows=[json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
A=torch.load('harmbench_activations/activations.pt',map_location='cpu')
H={r['pair_id']:A[r['pair_id']]['16'].float().numpy().astype(np.float64) for r in rows}
T=np.load('trait_matrix_layer16.npy').astype(np.float64)
Q=np.linalg.svd(T,full_matrices=False)[2].T
W=[18,73,30,29,71]; S=[11,3,32,77]
cell={(r['behavior_id'],r['jailbreak_idx']):r for r in rows}
behs=sorted({r['behavior_id'] for r in rows})
D=[];Y0=[];Y1=[]
for b in behs:
  for w in W:
    for s in S:
      D.append(H[cell[(b,s)]['pair_id']]-H[cell[(b,w)]['pair_id']]); Y0.append(cell[(b,w)]['jailbroken']); Y1.append(cell[(b,s)]['jailbroken'])
D=np.array(D); print('n pairs',len(D))
sh=lambda X,B: np.mean(np.sum((X@B)**2,1)/np.sum(X**2,1))
print('T share weak->strong', sh(D,Q))
print('median |PD|',np.median(np.linalg.norm(D@Q,axis=1)),'median |D|',np.median(np.linalg.norm(D,axis=1)))
rng=np.random.default_rng(0)
print('haar', np.mean([sh(D,np.linalg.qr(rng.standard_normal((4096,229)))[0]) for _ in range(3)]))
# covariance-drawn 229-dim subspaces
allH=np.array(list(H.values())); C=allH-allH.mean(0)
U,s,Vt=np.linalg.svd(C,full_matrices=False)
for k in [5]:
  print('PCA',k, sh(D,Vt[:k].T))
for _ in range(3):
  G=rng.standard_normal((229,C.shape[0]))@C  # samples from N(0,Sigma) (unnormalized)
  Qc=np.linalg.qr(G.T)[0]; print('cov-drawn 229', sh(D,Qc))
# random rotation within top K PCs, rank 229
for K in [400,1000,2000]:
  R=np.linalg.qr(rng.standard_normal((K,229)))[0]; print('rot in top',K, sh(D,Vt[:K].T@R))
print('y0 mean',np.mean(Y0),'y1',np.mean(Y1))
# dr->template
dr={r['behavior_id']:H[r['pair_id']] for r in rows if r['attack_type']=='direct_request'}
D2=np.array([H[r['pair_id']]-dr[r['behavior_id']] for r in rows if r['attack_type']=='human_jailbreak'])
print('DR->templ share',sh(D2,Q))
# split robustness of labels
asr={}
import collections
jb=collections.defaultdict(dict)
for r in rows:
  if r['attack_type']=='human_jailbreak': jb[r['jailbreak_idx']][r['behavior_id']]=r['jailbroken']
cnt=collections.Counter()
for seed in range(2000):
  rr=np.random.default_rng(seed); perm=rr.permutation(behs); Aset=set(perm[:80]); Bset=set(perm[80:])
  for half,name in [(Aset,'A'),(Bset,'B')]:
    w=tuple(sorted(t for t in jb if np.mean([jb[t][b] for b in half])<=0.10))
    s_=tuple(sorted(t for t in jb if np.mean([jb[t][b] for b in half])>=0.80))
    cnt[(w==tuple(sorted(W)) and s_==tuple(sorted(S)))]+=1
print('fraction of halves reproducing the 5/4 split', cnt[True]/sum(cnt.values()))
for t in [96,71,30,29,77,88]:
  vals=[]
  for seed in range(2000):
    perm=np.random.default_rng(seed).permutation(behs); vals.append(np.mean([jb[t][b] for b in perm[:80]]))
  vals=np.array(vals); print(t, 'P(<=.10)',np.mean(vals<=0.10),'P(>=.80)',np.mean(vals>=0.80))
