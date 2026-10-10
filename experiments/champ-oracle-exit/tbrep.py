import sys,duckdb,numpy as np,subprocess,json
O='runs/schema=combat_v4/date=2026-10-04/id=champ-ox-turnbench-r09/'
arms=sys.argv[1:]
c=duckdb.connect(); c.execute("set memory_limit='2GB'; set threads=1")
for a in ['baseline']+arms:
    f=f"read_parquet('{O}{a}/fights-*.parquet')"
    w,n=c.execute(f"select sum(won::int),count(*) from {f}").fetchone()
    df=c.execute(f"select sum(won::int),count(*) from {f} where list_contains([x.id for x in start.deck],107)").fetchone()
    sm=json.load(open(O+a+'/summary.json'))
    s=f"{a}: {w}/{n}={w/n:.3f} (±{1.96*(w/n*(1-w/n)/n)**.5:.3f}) DF {df[0]}/{df[1]} noDF {w-df[0]}/{n-df[1]} {sm['mean_seconds_per_fight']:.2f}s/f cap{sm['capped']}"
    if a=='baseline':
        h=[x[0] for x in c.execute(f"select turns_hist from read_parquet('{O}{a}/decisions_stats-*.parquet')").fetchall() if x[0]]
        L=max(map(len,h)); t=np.zeros(L)
        for x in h: t[:len(x)]+=x
        cs=np.cumsum(t)/t.sum(); s+=f" turns/sim {(t*np.arange(L)).sum()/t.sum():.2f}/p90 {int(np.argmax(cs>=.9))}"
    else:
        T=f"read_parquet('{O}{a}/turn_stats-*.parquet')"
        r=c.execute(f"select count(*),avg(fallback::int),avg(max_depth) filter(where not fallback),quantile_cont(max_depth,.9) filter(where not fallback),avg(mean_leaf_depth) filter(where not fallback),quantile_cont(mean_leaf_depth,.9) filter(where not fallback) from {T}").fetchone()
        s+=f"\n   turns {r[0]} fallback {r[1]*100:.1f}% | tree max-depth mean {r[2]:.2f} p90 {r[3]:.1f} | leaf depth mean {r[4]:.2f} p90 {r[5]:.2f}"
        s+="\n   fallback by reason: "+str(c.execute(f"select reason,count(*),round(100.0*count(*)/(select count(*) from {T}),1) from {T} where fallback group by 1").fetchall())
    print(s)
    if a!='baseline':
        out=subprocess.run(['.venv/bin/python','apps/pv/compare.py',O+'baseline',O+a],capture_output=True,text=True).stdout
        print('  ',[l for l in out.splitlines() if l.startswith(('| all','| demon','| no demon'))])
