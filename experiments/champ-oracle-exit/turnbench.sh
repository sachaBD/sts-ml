cd /home/sborowsk/project/sts_combat_rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=runs/schema=combat_v4/date=2026-10-04
O=$D/id=champ-ox-turnbench-r09; mkdir -p $O
W=$PWD/build/frozen/pv_worker.turn-search-v1-58c5c4ab8d3d
M=$D/id=champ-ox-c-r09/model/model.onnx
B=$D/id=champ-bench-nodome/out/fights-00000.parquet
R=experiments/champ-oracle-exit/RUNBOOK.md
run(){ n=$1; shift; echo "$(date -u +%FT%TZ) LAUNCH turnbench $n (1 worker, cpu11)" >> $R; taskset -c 11 .venv/bin/python apps/pv/play.py --agent pv --oracle --model $M --workers 1 --worker $W --starts $B --out $O/$n "$@" > $O/$n.log 2>&1 & pid=$!; echo $pid > $O/$n.pid; wait $pid; rc=$?; rm -f $O/$n.pid; echo "$(date -u +%FT%TZ) DONE turnbench $n rc=$rc" >> $R; [ $rc = 0 ] || exit $rc; touch $O/$n.done; }
run baseline --sims 800
run ts64 --turn-search --sims 64
run ts16 --turn-search --sims 16
run ts256 --turn-search --sims 256
