#!/usr/bin/env bash
# bench_paused.sh ID MODEL_DIR SIMS [play.py args]: pause running teacher workers (SIGSTOP), bench on the 409 nodome
# Champ starts with 9 workers, resume them.
cd "$(dirname "$0")/../.."
id=$1 model=$2 sims=$3; shift 3
pids=$(pgrep -f "pv_worker.q1 teacher" || true)
[ -n "$pids" ] && kill -STOP $pids
experiments/combat-pv/play.sh "$id" runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet \
  build/frozen/pv_worker.q1 pv "$sims" 9 --model "$model/model.onnx" "$@"
[ -n "$pids" ] && kill -CONT $pids
d=$(ls -d runs/schema=combat_v4/date=*/id=$id | tail -1)
PYTHONPATH=. .venv/bin/python apps/pv/compare.py runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out "$d/out" > "$d/compare.md" 2>&1
echo "$(date -u +%FT%TZ) BENCH $id $(grep -E '^\| all' $d/compare.md | tail -1)" >> experiments/combat-pv/RUNBOOK.md
