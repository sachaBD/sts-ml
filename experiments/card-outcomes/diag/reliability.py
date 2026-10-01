"""Diagnostic: seed split-half reliability of observed per-pair card effects (final HP, death=0) on dev predictions."""
import json,sys,collections,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1])]
syn=[r for r in rows if r['domain']=='synthetic']
print('variants',collections.Counter(r['variant'] for r in syn).most_common(6))
g=collections.defaultdict(dict)
for r in syn: g[(r['category'],r['cluster'],r['group_seed'],r['encounter'],r['seed'])][r['variant']]=(r['final_hp'] if r['won'] else 0, r['p_win']*r['mean_hp_if_win'])
base=min(collections.Counter(r['variant'] for r in syn))  # guess
pairs=collections.defaultdict(lambda: collections.defaultdict(list))
for (cat,cl,gs,enc,seed),v in g.items():
  b=v.get(0) if 0 in v else v.get('base')
  if b is None: continue
  for k,x in v.items():
    if k in (0,'base'): continue
    pairs[(cat,cl,gs,enc,k)][seed]=(x[0]-b[0], x[1]-b[1])
bycat=collections.defaultdict(list)
for key,s in pairs.items():
  seeds=sorted(s); 
  if len(seeds)<2: continue
  h=len(seeds)//2
  a=st.mean(s[x][0] for x in seeds[:h]); b=st.mean(s[x][0] for x in seeds[h:]); m=st.mean(s[x][0] for x in seeds)
  sd=st.pstdev([s[x][0] for x in seeds]); pm=st.mean(s[x][1] for x in seeds)
  bycat[key[0]].append((a,b,m,sd,len(seeds),pm))
for cat,L in bycat.items():
  A=[x[0] for x in L];B=[x[1] for x in L]
  try: c=st.correlation(A,B)
  except Exception: c=float('nan')
  M=[x[2] for x in L]; P=[x[5] for x in L]
  se=st.mean(x[3]/(x[4]-1)**.5 for x in L)
  try: cp=st.correlation(P,M)
  except Exception: cp=float('nan')
  print(f'{cat}: pairs={len(L)} seeds={L[0][4]} mean|obs delta|={st.mean(abs(m) for m in M):.2f} sd(obs delta)={st.pstdev(M):.2f} mean SE per pair={se:.2f} split-half r={c:.3f} corr(pred,obs)={cp:.3f} sd(pred)={st.pstdev(P):.2f}')
# Unbiased skill vs zero-effect: MSE(pred,obs)-MSE(0,obs) = mean(pred^2 - 2 pred obs); observation noise cancels.
# Negative = better than zero. SE by clustering on donor/group cluster.
print('--- unbiased dMSE vs zero (negative=better), cluster SE ---')
for cat in sorted(bycat):
  cl=collections.defaultdict(list)
  for key,s in pairs.items():
    if key[0]!=cat: continue
    for x in s.values(): cl[key[1]].append(x[1]**2-2*x[1]*x[0])
  per=[sum(v) for v in cl.values()]; n=sum(len(v) for v in cl.values())
  est=sum(per)/n; k=len(per)
  se=(k/(k-1))**.5*(sum((p-est*len(v))**2 for p,v in zip(per,cl.values())))**.5/n if k>1 else float('nan')
  print(f'{cat}: dMSE={est:+.3f} ± {se:.3f} (clusters={k}, obs={n})')
