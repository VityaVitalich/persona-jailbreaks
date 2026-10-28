import json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
rng=np.random.default_rng(3)
T=np.load('trait_matrix_layer16.npy').astype(np.float64); Q=np.linalg.svd(T,full_matrices=False)[2].T
A=torch.load('harmbench_activations/activations.pt',weights_only=False)
R=[json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
H={r['pair_id']:A[r['pair_id']]['16'].float().numpy().astype(np.float64) for r in R}
cell={(r['jailbreak_idx'],r['behavior_id']):r['pair_id'] for r in R}
behs=sorted({r['behavior_id'] for r in R}); rng.shuffle(behs); Ab=set(behs[:80]); Bb=behs[80:]
W=[18,73,30,29,71]; S=[11,3,32,77]
Xa=np.stack([H[r['pair_id']] for r in R if r['behavior_id'] in Ab]); mu=Xa.mean(0); U,s,Vt=np.linalg.svd(Xa-mu,full_matrices=False)
h0=np.stack([H[cell[(t0,b)]] for b in Bb for t0 in W for t1 in S]); h1=np.stack([H[cell[(t1,b)]] for b in Bb for t0 in W for t1 in S]); D=h1-h0
def res(Z,k): z=(Z-mu)@Vt[:k].T; return np.median((np.sum((Z-mu)**2,1)-np.sum(z**2,1)))
Qh=np.linalg.qr(rng.standard_normal((4096,229)))[0]
Ke=740; Rk=np.linalg.qr(rng.standard_normal((Ke,229)))[0]; Be=Vt[:Ke].T@Rk
for k in [200,500,1000]:
  print(k, 'h0',round(res(h0,k),2),'h1',round(res(h1,k),2),'h*T',round(res(h0+D@Q@Q.T,k),2),'h*haar',round(res(h0+D@Qh@Qh.T,k),2),'h*energy(top740 rot)',round(res(h0+D@Be@Be.T,k),2), 'h*compl', round(res(h1-D@Q@Q.T,k),2))
