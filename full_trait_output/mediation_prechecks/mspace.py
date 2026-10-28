import json, glob, os, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
os.chdir('/Users/viktor/persona-vector-jbing/full_trait_output')
rng=np.random.default_rng(0)
def orth(M):  # rows span -> orthonormal basis columns
    U,s,Vt=np.linalg.svd(M.astype(np.float64),full_matrices=False); return Vt.T  # [d,k]
T=np.load('trait_matrix_layer16.npy').astype(np.float64); Q=orth(T); d=T.shape[1]; k=Q.shape[1]
# 240 no-filter set at L16
nfV=[torch.load(f,weights_only=False)['vector'][16].float().numpy() for f in sorted(glob.glob('traits40_vectors/pre_generation_last_token/all_traits_no_filter/*.pt'))]
Q240=orth(np.stack(nfV))
# HarmBench L16
A=torch.load('harmbench_activations/activations.pt',weights_only=False)
R=[json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
pid=[r['pair_id'] for r in R]; H=np.stack([A[p]['16'].float().numpy() for p in pid]).astype(np.float64)
y=np.array([bool(r['jailbroken']) for r in R]); at=np.array([r['attack_type'] for r in R]); beh=np.array([r['behavior_id'] for r in R]); tm=np.array([r['jailbreak_idx'] for r in R])
print('HB n',len(H),'norm median',np.median(np.linalg.norm(H,axis=1)), 'DR median',np.median(np.linalg.norm(H[at=='direct_request'],axis=1)),'HJ median',np.median(np.linalg.norm(H[at=='human_jailbreak'],axis=1)))
mu=H.mean(0); print('||global mean||',np.linalg.norm(mu), ' mean share of ||h||^2 explained by mean dir', np.mean((H@mu/np.linalg.norm(mu))**2/np.sum(H**2,1)))
# PCA bases: in-sample, and held-out by behavior split
Hc=H-mu
def pca_basis(X,kk): U,s,Vt=np.linalg.svd(X-X.mean(0),full_matrices=False); return Vt[:kk].T
ub=np.unique(beh); trb=set(rng.choice(ub,len(ub)//2,replace=False)); trm=np.array([b in trb for b in beh]); tem=~trm
Qp_in=pca_basis(H,k); Qp_tr=pca_basis(H[trm],k)
Qr=[np.linalg.qr(rng.standard_normal((d,k)))[0] for _ in range(10)]
def frac(V,B):  # mean of per-vector ratio, and ratio of sums
    V=np.atleast_2d(V); P=V@B; num=np.sum(P**2,1); den=np.sum(V**2,1); return np.mean(num/den), num.sum()/den.sum()
def row(name,V,mask=None):
    out=[name, len(np.atleast_2d(V))]
    for B in [Q,Q240]: out.append(frac(V,B)[0])
    out.append(np.mean([frac(V,B)[0] for B in Qr]))
    out.append(frac(V,Qp_in)[0])
    out.append(frac(V if mask is None else V[mask],Qp_tr)[0] if mask is not None else np.nan)
    print('| '+' | '.join([out[0],str(out[1])]+[f'{x:.3f}' for x in out[2:]])+' |')
print('| quantity | n | trait229 | trait240(nofilt) | random229 | PCA229 in-sample | PCA229 held-out beh |')
row('h (raw)',H,tem); row('h - global mean',Hc,tem)
ia=rng.integers(0,len(H),5000); ib=rng.integers(0,len(H),5000); D=H[ia]-H[ib]; ok=np.linalg.norm(D,axis=1)>0
row('h_a-h_b random pairs',D[ok],(tem[ia]&tem[ib])[ok])
hj=at=='human_jailbreak'
row('mean(jb)-mean(ref) [HJ]',H[hj&y].mean(0)-H[hj&~y].mean(0))
row('mean(HJ)-mean(DR)',H[hj].mean(0)-H[at=='direct_request'].mean(0))
# same-behavior DR -> template pairs
dr={b:H[i] for i,(b,a) in enumerate(zip(beh,at)) if a=='direct_request'}
P1=[(H[i]-dr[beh[i]],tem[i],y[i]) for i in range(len(H)) if hj[i] and beh[i] in dr]
Dp=np.stack([p[0] for p in P1]); mp=np.array([p[1] for p in P1]); yp=np.array([p[2] for p in P1])
row('h1-h0 same beh, DR->HJ template (all)',Dp,mp); row('  ... only where template jb=1',Dp[yp],mp[yp])
# within behavior refused-template -> jb-template
pairs=[]
for b in np.unique(beh):
    i1=np.where((beh==b)&hj&y)[0]; i0=np.where((beh==b)&hj&~y)[0]
    for a1 in i1[:3]:
        for a0 in i0[:3]: pairs.append(H[a1]-H[a0])
row('h(jb tmpl)-h(ref tmpl) same beh',np.stack(pairs))
# per-template family mean minus DR mean (HB, matched behaviours)
fam=[]
for t in np.unique(tm[hj]):
    m=hj&(tm==t); fam.append(H[m].mean(0)-H[at=='direct_request'].mean(0))
row('family mean(template t)-mean(DR), 20 tmpl',np.stack(fam))
# other attack families: matched to HB DR by behavior_id
for f in ['gcg','pair','pap','pez','gptfuzz']:
    a=torch.load(f'{f}_activations/activations.pt',weights_only=False); rr=[json.loads(l) for l in open(f'{f}_activations/responses.jsonl')]
    X=[];Y=[]
    for r in rr:
        if r['behavior_id'] in dr: X.append(a[r['pair_id']]['16'].float().numpy().astype(np.float64)-dr[r['behavior_id']] if r['pair_id'] in a else None); Y.append(str(r['jailbroken'])=='True')
    keep=[i for i,x in enumerate(X) if x is not None]; X=np.stack([X[i] for i in keep]); Y=np.array([Y[i] for i in keep])
    allX=np.stack([a[r['pair_id']]['16'].float().numpy() for r in rr if r['pair_id'] in a]); asr=np.mean([str(r['jailbroken'])=='True' for r in rr])
    row(f'{f}: h1-h0 same beh vs DR (ASR {asr:.2f}, matched {len(X)})',X)
    row(f'{f}: family mean - DR mean',allX.mean(0)-H[at=='direct_request'].mean(0))
# patch magnitude in steering units: ||P_T(h1-h0)|| for DR->HJ pairs
pn=np.linalg.norm(Dp@Q,axis=1); fn=np.linalg.norm(Dp,axis=1)
print('DR->HJ: median ||h1-h0||',np.median(fn),' median ||P_T(h1-h0)||',np.median(pn),' ratio',np.median(pn/fn),' in RESIDUAL_NORM=22 units (alpha):',np.median(pn)/22)
print('median ||h0|| DR',np.median(np.linalg.norm(np.stack(list(dr.values())),axis=1)))
# low-rank trait subspaces
U,s,Vt=np.linalg.svd(T,full_matrices=False)
for kk in [5,10,30,58,96]:
    B=Vt[:kk].T; print(f'trait top-{kk} SVD subspace: frac DR->HJ pair diff {frac(Dp,B)[0]:.3f}, random-{kk} expect {kk/d:.4f}, PCA-{kk} in-sample {frac(Dp,Qp_in[:,:kk])[0]:.3f}, h-mu {frac(Hc,B)[0]:.3f}')
# overlap trait span vs PCA span
sv=np.linalg.svd(Q.T@Qp_in,compute_uv=False); print('trait229 vs PCA229 principal cosines: mean cos^2',np.mean(sv**2),' #cos>0.9',(sv>0.9).sum(),' #cos>0.5',(sv>0.5).sum())
# how much of the trait span the activations actually use: variance of centered HB in trait span by trait-SVD direction
ev=np.var(Hc@Q,0).sum(); print('total var of centered HB',np.var(Hc,0).sum(),' in trait span',ev,' frac',ev/np.var(Hc,0).sum())
