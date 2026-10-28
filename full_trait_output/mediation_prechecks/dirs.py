import json, numpy as np, torch, warnings, os
warnings.filterwarnings('ignore'); os.chdir('/Users/viktor/persona-vector-jbing/full_trait_output')
T=np.load('trait_matrix_layer16.npy').astype(np.float64); Q=np.linalg.svd(T,full_matrices=False)[2].T
U,s,Vt=np.linalg.svd(T,full_matrices=False)
def share(v,B=Q): v=np.asarray(v,np.float64); v=v/np.linalg.norm(v); p=v@B; return float(p@p)
def maxcos(v): v=v/np.linalg.norm(v); c=T@v; i=np.argmax(np.abs(c)); return c[i], i
names=json.load(open('trait_names_layer16.json'))
dirs={}
for f in ['steering_directions_layer16','control_directions_layer16','ablation_directions_layer16']:
    d=torch.load(f'steering_directions/{f}.pt',weights_only=False,map_location='cpu')['vectors']
    for k,v in d.items(): dirs[f.split('_')[0]+':'+k]=v.double().numpy()
md=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/mean_diffs.pt',weights_only=False,map_location='cpu').numpy()
r1=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt',weights_only=False,map_location='cpu')['direction'].numpy()
r5=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16.pt',weights_only=False,map_location='cpu')['direction'].numpy()
print('r1 == mean_diffs[-1,16]?', np.allclose(r1,md[4,16]), ' r5 == mean_diffs[0,16]?', np.allclose(r5,md[0,16]))
dirs['arditi r pos-1 blockin16 (=our L15)']=md[4,16]; dirs['arditi r pos-1 blockin17 (=our L16)']=md[4,17]
dirs['arditi r pos-5 blockin16']=md[0,16]; dirs['arditi r pos-5 blockin17']=md[0,17]
dirs['arditi selected (pos-2, blockin11)']=torch.load('/Users/viktor/refusal_direction/runs/llama-3.1-8b-instruct/direction.pt',weights_only=False,map_location='cpu').numpy()
# HB global mean direction
A=torch.load('harmbench_activations/activations.pt',weights_only=False,map_location='cpu')
H=np.stack([v['16'].float().numpy() for v in A.values()]).astype(np.float64); dirs['HB global mean h (L16)']=H.mean(0)
print('| direction | share in span(229) | top-10 trait-SVD share | top-30 share | max |cos| single trait |')
for k,v in dirs.items():
    c,i=maxcos(v); print(f'| {k} | {share(v):.3f} | {share(v,Vt[:10].T):.3f} | {share(v,Vt[:30].T):.3f} | {abs(c):.3f} ({names[i]}) |')
print('random expectation', 229/4096)
# Arditi r at each block-input layer, pos -1: share in L16 trait span (diagnostic only)
print('arditi pos-1 share by block-input layer', {l:round(share(md[4,l]),3) for l in range(8,25)})
