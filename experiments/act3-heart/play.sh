#!/usr/bin/env bash
# One managed Heart-mode play job (runs.run, schema overworld_v1) with the frozen worker.
#   play.sh NAME CKPT FIRST_SEED SEEDS EPS ROUTE_P [extra play.py args...]
# Records overworld_v1 tables and combat_v4 fights ({out}/combat). Resume: rerun unchanged after deleting the
# failed run dir (runs.run refuses existing ids), or call play.py directly on the same --out.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NAME=$1 CKPT=$2 FIRST=$3 SEEDS=$4 EPS=$5 ROUTE=$6; shift 6
WORKER=build/overnight/frozen/run_rl_worker
echo "$(date -u +%FT%TZ) START $NAME ckpt=$CKPT seeds=$FIRST+$SEEDS eps=$EPS route=$ROUTE extra=$*" >> experiments/act3-heart/RUNBOOK.md
.venv/bin/python -m runs.run overworld_v1 "$NAME" --no-compact --input "$CKPT" --input "$WORKER" -- \
  .venv/bin/python apps/run_rl/play.py --out '{out}' --first-seed "$FIRST" --seeds "$SEEDS" --workers 10 \
  --worker "$WORKER" --policy net --ckpt "$CKPT" --max-act 4 --target full \
  --decide rest path shop event neow boss_relic --sims 500,5000,10000,10000,20000 \
  --eps "$EPS" --route-p "$ROUTE" --overworld-record --collection-id "$NAME" --combat-out '{out}/combat' "$@"
echo "$(date -u +%FT%TZ) DONE $NAME" >> experiments/act3-heart/RUNBOOK.md
