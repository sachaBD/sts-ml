import sys,json,duckdb,numpy as np
r=int(sys.argv[1]); tag=sys.argv[2] if len(sys.argv)>2 else 'c'
R=f'runs/schema=combat_v4/date=2026-10-04/id=champ-ox-{tag}-r{r:02d}/'
c=duckdb.connect(); c.execute("set memory_limit='2GB'; set threads=2")
def S(p): return json.load(open(R+p+'/summary.json'))
sp=S('play'); out={'sp':f"{sp['wins']}/{sp['fights']}={sp['wins']/sp['fights']:.3f} cap{sp['capped']} {sp['wall_seconds']/60:.1f}min"}
v=[json.loads(l) for l in open(R+'train.log') if l.startswith('{"epoch"')]
out['train']=' | '.join(f"e{x['epoch']} val v{x['val']['value_loss']:.3f} ce{x['val']['policy_loss']:.3f} top1{x['val']['top1']:.3f} (train v{x['train']['value_loss']:.3f})" for x in v)
def bench(a):
    f=f'{R}{a}/fights-*.parquet'
    w,n=c.execute(f"select sum(won::int),count(*) from read_parquet('{f}')").fetchone()
    df=c.execute(f"select sum(won::int),count(*) from read_parquet('{f}') where list_contains([x.id for x in start.deck],107)").fetchone()
    h=[x[0] for x in c.execute(f"select turns_hist from read_parquet('{R}{a}/decisions_stats-*.parquet')").fetchall() if x[0]]
    t=''
    if h:
        L=max(map(len,h)); s=np.zeros(L)
        for x in h: s[:len(x)]+=x
        cs=np.cumsum(s)/s.sum(); t=f" turns {(s*np.arange(L)).sum()/s.sum():.2f}/p90 {int(np.argmax(cs>=.9))}"
    sm=S(a)
    return f"{a} {w}/{n}={w/n:.3f} DF {df[0]}/{df[1]} noDF {w-df[0]}/{n-df[1]} {sm['mean_seconds_per_fight']:.2f}s/f cap{sm['capped']}{t}", w/n
import os
for a in ['bench-policy','bench-oracle','bench-real']:
    if os.path.exists(R+a+'/summary.json'): s,p=bench(a); out[a]=s; out[a+'_p']=p
print(f"| r{r:02d} | selfplay {out['sp']} | {out['train']} | "+' | '.join(out[a] for a in ['bench-policy','bench-oracle','bench-real'] if a in out)+' |')
al=[]
if out['bench-policy_p']<.17: al.append('policy<17')
if out['bench-oracle_p']<.22: al.append('oracle800<22')
if out.get('bench-real_p',0)>=.46: al.append('real>=46')
if out.get('bench-real_p',1)<.18: al.append('real<18')
if v and v[-1]['val']['value_loss']>.15: al.append('valvalue>.15')
if sp['capped']/sp['fights']>.01: al.append('capped')
print('ALERTS:',al)
