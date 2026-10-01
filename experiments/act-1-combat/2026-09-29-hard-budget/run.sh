#!/usr/bin/env bash
# Run this study's configs in order, one at a time; stop at the first failure.
#   experiments/act-1-combat/2026-09-29-hard-budget/run.sh easy-cheap-mcts-20k-s0 easy-cheap-mcts-10-s0 ...     (ids = configs/<id>.toml)
# Per-run log: experiments/act-1-combat/2026-09-29-hard-budget/logs/<id>.log. Prints one line per run (start/finish/wall/skipped).
set -u
cd "$(dirname "$0")/../../.."
D=experiments/act-1-combat/2026-09-29-hard-budget
mkdir -p $D/logs
for id in "$@"; do
  cfg=$D/configs/$id.toml
  [ -f "$cfg" ] || { echo "$(date -u +%H:%MZ) MISSING $cfg"; exit 1; }
  if ls -d runs/schema=combat_v3/date=*/id=$id >/dev/null 2>&1; then
    if grep -q '"status": "done"' runs/schema=combat_v3/date=*/id=$id/run.json; then echo "$(date -u +%H:%MZ) SKIP $id (already done)"; continue; fi
    echo "$(date -u +%H:%MZ) FAIL $id: an unfinished run with this id exists (rerun as a copy with id $id-r2)"; exit 1
  fi
  start=$(date +%s); echo "$(date -u +%H:%MZ) START $id (load $(cut -d" " -f1-3 /proc/loadavg))"
  if ./apps/value_play/run.sh "$cfg" > $D/logs/$id.log 2>&1; then
    s=$(grep -E "skipped:|seconds/fight:" $D/logs/$id.log | tr -s ' ' | tr '\n' ' ')
    echo "$(date -u +%H:%MZ) DONE $id ($(( ($(date +%s)-start)/60 )) min) $s"
  else
    echo "$(date -u +%H:%MZ) FAIL $id (see $D/logs/$id.log)"; tail -5 $D/logs/$id.log; exit 1
  fi
done
