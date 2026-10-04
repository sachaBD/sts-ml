cd /home/sborowsk/project/sts_combat_rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
W=build/frozen/pv_worker.ox-e902f9a
M=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-iteration-0/model/model.onnx
B=runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet
T=runs/schema=combat_v4/date=2026-10-03/id=champ-bench-teacher20k/out
SW=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-sweep0
mkdir -p $SW
R=experiments/champ-oracle-exit/RUNBOOK.md
run(){ n=$1; shift; echo "$(date -u +%FT%TZ) LAUNCH sweep0 $n" >> $R; "$@" > "$SW/$n.log" 2>&1 & pid=$!; echo $pid > "$SW/$n.pid"; wait $pid; rc=$?; rm "$SW/$n.pid"; echo "$(date -u +%FT%TZ) DONE sweep0 $n rc=$rc" >> $R; test $rc = 0 || exit $rc; }
P="$PY apps/pv/play.py --starts $B --agent pv --model $M --workers 9 --worker $W"
run policy $P --sims 1 --policy-only --out $SW/policy
run oracle200 $P --sims 200 --oracle --out $SW/oracle200
run oracle800 $P --sims 800 --oracle --out $SW/oracle800
run real2000 $P --sims 2000 --out $SW/real2000
run oracle3200 $P --sims 3200 --oracle --out $SW/oracle3200
$PY apps/pv/compare.py $T $SW/policy $SW/oracle200 $SW/oracle800 $SW/real2000 $SW/oracle3200 > $SW/compare.md 2>&1
echo ALLDONE > $SW/sweep.done
