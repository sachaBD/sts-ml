cd /home/sborowsk/project/sts_combat_rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
E=experiments/champ-oracle-exit; D4=runs/schema=combat_v4/date=2026-10-04; N=runs/schema=combat_v4/date=2026-10-05
W=build/frozen/pv_worker.turn-targets-dc8f9579d2b6; B=$D4/id=champ-bench-nodome/out/fights-00000.parquet
INIT=$D4/id=champ-ox-c-r18/model/model.pt
C="$D4/id=champ-ox-c-r16/rows/rows.parquet $D4/id=champ-ox-c-r17/rows/rows.parquet $D4/id=champ-ox-c-r18/rows/rows.parquet"
D="$N/id=champ-ox-d-r01/rows/rows.parquet $N/id=champ-ox-d-r02/rows/rows.parquet $N/id=champ-ox-d-r03/rows/rows.parquet"
L(){ echo "$(date -u +%FT%TZ) $*" >> $E/RUNBOOK.md; }
mk(){ rm -f "$1".pid; }
rc_run(){ # name log cmd...
  n=$1; lg=$2; shift 2; "$@" > $lg 2>&1 & p=$!; echo $p > $N/$n.pid; wait $p; rc=$?; rm -f $N/$n.pid; L "DONE diag $n rc=$rc"; return $rc; }
# train chain in background
( for k in 1 2 3 4 5; do
  case $k in 1) R="$C"; X="";; 2) R="$D"; X="";; 3) R="$D"; X="--policy-weight 0";; 4) R="$D"; X="--value-weight 0";; 5) R="$C $D"; X="";; esac
  O=$N/id=champ-diag-D$k; mkdir -p $O; L "LAUNCH diag D$k train $X"
  rc_run D$k-train $O/train.log .venv/bin/python -m agents.combat.pv.train --data $R --out $O/model --init $INIT --device cuda --stream --epochs 1 --value-mix 0.5 --grad-clip 1.0 $X || exit 1
  touch $O/train.done
done ) &
# E64 oracle diag first (CPU), then per-model benches
O0=$N/id=champ-diag-c18-ts64; mkdir -p $O0; L "LAUNCH diag c18-ts64"
rc_run c18ts64 $O0.log .venv/bin/python apps/pv/play.py --agent pv --model $D4/id=champ-ox-c-r18/model/model.onnx --oracle --turn-search --sims 64 --workers 9 --worker $W --starts $B --out $O0 || exit 1
touch $O0.done
for k in 1 2 3 4 5; do
  O=$N/id=champ-diag-D$k
  while [ ! -e $O/train.done ]; do sleep 15; done
  L "LAUNCH diag D$k bench-real2000"
  rc_run D$k-real $O/bench-real.log .venv/bin/python apps/pv/play.py --agent pv --model $O/model/model.onnx --sims 2000 --workers 9 --worker $W --starts $B --out $O/bench-real || exit 1
  rc_run D$k-pol $O/bench-policy.log .venv/bin/python apps/pv/play.py --agent pv --model $O/model/model.onnx --sims 1 --policy-only --workers 9 --worker $W --starts $B --out $O/bench-policy || exit 1
  touch $O/bench.done
done
touch $N/diag.done
