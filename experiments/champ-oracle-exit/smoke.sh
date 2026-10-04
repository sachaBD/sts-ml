cd /home/sborowsk/project/sts_combat_rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
W=build/frozen/pv_worker.ox-v3-precommit
M=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-iteration-0/model/model.onnx
SM=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-smoke
run(){ n=$1; shift; "$@" > "$SM/$n.log" 2>&1 & pid=$!; echo $pid > "$SM/$n.pid"; wait $pid; rc=$?; rm "$SM/$n.pid"; echo "$(date -u +%FT%TZ) DONE smoke $n rc=$rc" >> experiments/champ-oracle-exit/RUNBOOK.md; test $rc = 0 || exit $rc; }
run oracle800 $PY apps/pv/play.py --starts "$SM/starts.parquet" --agent pv --model "$M" --sims 800 --oracle --workers 9 --worker "$W" --out "$SM/oracle800"
run real2000 $PY apps/pv/play.py --starts "$SM/starts.parquet" --agent pv --model "$M" --sims 2000 --workers 9 --worker "$W" --out "$SM/real2000"
for arm in oracle800 real2000; do
 run $arm-replay $PY -m agents.combat.pv.data --fights "$SM/$arm"/fights-*.parquet --search "$SM/$arm"/search-*.parquet --encounters 39 --out "$SM/$arm-replay" --worker "$W"
done
echo ALLDONE > $SM/smoke.done
