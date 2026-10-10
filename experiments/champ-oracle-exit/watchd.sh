cd /home/sborowsk/project/sts_combat_rl; E=experiments/champ-oracle-exit; export DATE=2026-10-05
r=${1:-1}
while [ $r -le 9 ]; do
  R=runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r$(printf %02d $r)
  last=bench-policy.done; [ $((r%3)) = 0 ] && last=compare.done
  while [ ! -e $R/$last ]; do kill -0 $(cat $E/exit-d.pid) 2>/dev/null || { echo DRIVERDEAD r$r; exit 1; }; sleep 20; done
  .venv/bin/python $E/rowc.py $r d > /tmp/d_r$r 2>&1; head -1 /tmp/d_r$r >> $E/LOG.md
  # time-cap fallbacks (>1% of turns) over play + bench-oracle
  TC=$(.venv/bin/python - <<PY
import duckdb
R='$R/'
c=duckdb.connect(); c.execute("set memory_limit='2GB'; set threads=1")
t=f=0
for a in ['play','bench-oracle']:
    try:
        n,k=c.execute(f"select count(*),sum((reason ilike '%time%' or time_overshoot)::int) from read_parquet('{R}{a}/turn_stats-*.parquet')").fetchone(); t+=n; f+=k or 0
    except Exception as e: pass
print(f"{(f/t*100 if t else 0):.2f}")
PY
)
  echo "r$r timecap% $TC"; cat /tmp/d_r$r
  [ "$(awk -v x=$TC 'BEGIN{print (x>1)}')" = 1 ] && { echo "ALERT timecap"; exit 2; }
  grep -q 'ALERTS: \[.*[a-z]' /tmp/d_r$r && { echo ALERT; exit 2; }
  [ $((r%3)) = 0 ] && exit 0
  r=$((r+1))
done
