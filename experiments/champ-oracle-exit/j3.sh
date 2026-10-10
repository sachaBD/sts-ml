cd /home/sborowsk/project/sts_combat_rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
E=experiments/champ-oracle-exit; N=runs/schema=combat_v4/date=2026-10-05; D4=runs/schema=combat_v4/date=2026-10-04
S=runs/schema=pv_starts_v1/id=champ-bench2k/starts.parquet; W=build/frozen/pv_worker.turn-targets-dc8f9579d2b6
run(){ n=$1; shift; echo "$(date -u +%FT%TZ) LAUNCH J3 $n 9 workers" >> $E/RUNBOOK.md; "$@" > $N/id=champ-bench2k-$n.log 2>&1 & p=$!; echo $p > $N/id=champ-bench2k-$n.pid; wait $p; rc=$?; rm -f $N/id=champ-bench2k-$n.pid; echo "$(date -u +%FT%TZ) DONE J3 $n rc=$rc" >> $E/RUNBOOK.md; [ $rc = 0 ] || exit $rc; }
run teacher .venv/bin/python apps/pv/play.py --starts $S --agent teacher --sims 20000 --workers 9 --worker build/frozen/pv_worker.t1 --out $N/id=champ-bench2k-teacher
run c18 .venv/bin/python apps/pv/play.py --starts $S --agent pv --model $D4/id=champ-ox-c-r18/model/model.onnx --sims 2000 --workers 9 --worker $W --out $N/id=champ-bench2k-c18
run d5 .venv/bin/python apps/pv/play.py --starts $S --agent pv --model $N/id=champ-diag-D5/model/model.onnx --sims 2000 --workers 9 --worker $W --out $N/id=champ-bench2k-d5
.venv/bin/python apps/pv/compare.py $N/id=champ-bench2k-teacher $N/id=champ-bench2k-c18 $N/id=champ-bench2k-d5 > $N/id=champ-bench2k-compare-vs-teacher.md 2>&1
.venv/bin/python apps/pv/compare.py $N/id=champ-bench2k-c18 $N/id=champ-bench2k-d5 > $N/id=champ-bench2k-compare-d5-vs-c18.md 2>&1
touch $N/j3.done
