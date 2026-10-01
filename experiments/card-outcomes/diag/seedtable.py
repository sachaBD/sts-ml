"""Aggregate supp-eval checkpoints over training seeds: mean (min..max) per metric. Outputs markdown."""
import json,glob,subprocess,sys,statistics as st,re
def load(mid):
    d=glob.glob(f'runs/schema=combat_outcome_v1/date=*/id={mid}/out')[0]
    r=json.load(open(d+'/report.json'))['results']; out={}
    for arm in ('natural','augmented'):
        m=r[arm]['metrics']
        for k in ('natural/boss','natural/elite','natural/hard','synthetic/boss','synthetic/elite','synthetic/hard','synthetic/easy'):
            out[(arm,'Brier '+k)]=m[k]['brier']
        txt=subprocess.run([sys.executable,'experiments/card-outcomes/diag/reliability.py',f'{d}/{arm}-predictions.jsonl'],capture_output=True,text=True).stdout
        for cat,v in re.findall(r'^(\w+): dMSE=([-+.\d]+)',txt,re.M): out[(arm,'card dMSE '+cat)]=float(v)
    return out
stages=['s1','s2','s3','s3c']; seeds=['','-seed1','-seed2']
res={s:[load(f'card-outcomes-{s}-model-supp{x}') for x in seeds] for s in stages}
keys=list(res['s1'][0])
print('| arm | metric | '+' | '.join(stages)+' |\n|---|---|'+'---|'*len(stages))
for k in keys:
    cells=[]
    for s in stages:
        v=[r[k] for r in res[s]]; cells.append(f'{st.mean(v):.3f} ({min(v):.3f}..{max(v):.3f})' if 'dMSE' not in k[1] else f'{st.mean(v):+.2f} ({min(v):+.2f}..{max(v):+.2f})')
    print(f'| {k[0]} | {k[1]} | '+' | '.join(cells)+' |')
