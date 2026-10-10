cd /home/sborowsk/project/sts_combat_rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
E=experiments/champ-oracle-exit; D=runs/schema=combat_v4/date=2026-10-04; N=runs/schema=combat_v4/date=2026-10-05
M=$D/id=champ-ox-c-r15/model; B=$D/id=champ-bench-nodome/out/fights-00000.parquet
J1=$N/id=champ-pimc-r15-k4-e16; mkdir -p $J1
echo "$(date -u +%FT%TZ) LAUNCH J1 PIMC k4 e16 M=r15 9 workers $J1" >> $E/RUNBOOK.md
.venv/bin/python apps/pv/play.py --agent pv --model $M/model.onnx --turn-search --particles 4 --sims 16 --workers 9 --worker build/frozen/pv_worker.pimc-136e4ebe28f7 --starts $B --out $J1/out > $J1/play.log 2>&1 & pid=$!; echo $pid > $J1/play.pid; wait $pid; rc=$?; rm -f $J1/play.pid
echo "$(date -u +%FT%TZ) DONE J1 rc=$rc" >> $E/RUNBOOK.md
[ $rc = 0 ] && .venv/bin/python apps/pv/compare.py $D/id=champ-ox-c-r15/bench-real $J1/out > $J1/compare.md 2>&1
touch $J1/j1.done
