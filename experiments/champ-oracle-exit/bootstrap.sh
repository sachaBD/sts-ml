#!/usr/bin/env bash
# Encode teacher Champ fights under pv_champ_win_v3, then train a fresh iteration-0 model.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-iteration-0
mkdir -p "$run" build/frozen
echo $$ > "$run/bootstrap.pid"
trap 'rm -f "$run/bootstrap.pid"' EXIT
worker=build/frozen/pv_worker.ox-bootstrap-$(date -u +%Y%m%dT%H%M%S)-$$
cp build/pv/agents/combat/pv/pv_worker "$worker"
log() { echo "$(date -u +%FT%TZ) $*" >> experiments/champ-oracle-exit/RUNBOOK.md; }
for source in champ-train champ-teachergen-a; do
    if [[ ! -f "$run/$source/rows.parquet" ]]; then
        src=runs/schema=combat_v4/date=2026-10-04/id=$source/out
        .venv/bin/python -m agents.combat.pv.data --fights "$src"/fights-*.parquet --search "$src"/search-*.parquet \
            --encounters 39 --worker "$worker" --out "$run/$source"
        log "DONE iteration-0 encode $source contract=pv_champ_win_v3"
    fi
done
if [[ ! -f "$run/train.done" ]]; then
    .venv/bin/python -m agents.combat.pv.train --data "$run/champ-train/rows.parquet" "$run/champ-teachergen-a/rows.parquet" \
        --out "$run/model" --width 64 --epochs 10 --device cuda --stream
    touch "$run/train.done"
    log "DONE iteration-0 train width=64 epochs=10 fresh-init best-val-selection value=MSE target=100*won policy=teacher-visits log=$run/bootstrap.log"
fi
