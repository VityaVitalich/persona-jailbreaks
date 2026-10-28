import json,glob,numpy as np,torch,warnings,os
warnings.filterwarnings('ignore'); os.chdir('/Users/viktor/persona-vector-jbing/full_trait_output')
V=np.stack([torch.load(f,weights_only=False)['vector'].float().numpy() for f in sorted(glob.glob('traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/*.pt'))]) # [229,32,4096]
b=torch.load('all_layers_activations/harmbench_activations.pt',weights_only=False)
R=[json.loads(l) for l in open('all_layers_activations/harmbench_rows.jsonl')]
R=[r for r in R if r['pair_id'] in b]
beh=np.array([r['behavior_id'] for r in R]); at=np.array([r['attack_type'] for r in R]); y=np.array([bool(r['jailbroken']) for r in R])
dr_idx={beh[i]:i for i in range(len(R)) if at[i]=='direct_request'}
pairs=[(dr_idx[beh[i]],i) for i in range(len(R)) if at[i]=='human_jailbreak' and beh[i] in dr_idx]
print('n pairs',len(pairs))
print('| layer | rank | cond | share ||h-mu||^2 | share DR->HJ diff | share jb-ref mean | median ||h|| | median ||h1-h0|| |')
for L in range(0,32,2):
    TL=V[:,L,:].astype(np.float64); TL/=np.linalg.norm(TL,axis=1,keepdims=True)
    s=np.linalg.svd(TL,compute_uv=False); Q=np.linalg.svd(TL,full_matrices=False)[2].T
    H=np.stack([b[r['pair_id']][str(L)].float().numpy() for r in R]).astype(np.float64)
    Hc=H-H.mean(0); D=np.stack([H[j]-H[i] for i,j in pairs])
    f=lambda X: np.mean(np.sum((X@Q)**2,1)/np.sum(X**2,1))
    hj=at=='human_jailbreak'; m=H[hj&y].mean(0)-H[hj&~y].mean(0)
    print(f'| {L} | {np.sum(s>1e-6*s[0])} | {s[0]/s[-1]:.0f} | {f(Hc):.3f} | {f(D):.3f} | {f(m[None]):.3f} | {np.median(np.linalg.norm(H,axis=1)):.2f} | {np.median(np.linalg.norm(D,axis=1)):.2f} |')
