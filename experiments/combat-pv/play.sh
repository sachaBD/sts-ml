#!/usr/bin/env bash
# play.sh ID STARTS WORKER AGENT SIMS WORKERS [play.py extra args]  → runs/schema=combat_v4/date=<utc>/id=ID
set -euo pipefail
cd "$(dirname "$0")/../.."
id=$1 starts=$2 worker=$3 agent=$4 sims=$5 workers=$6; shift 6
echo "$(date -u +%FT%TZ) START play $id starts=$starts worker=$worker agent=$agent sims=$sims workers=$workers $*" >> experiments/combat-pv/RUNBOOK.md
.venv/bin/python -m runs.run combat_v4 "$id" --no-compact -- .venv/bin/python apps/pv/play.py --starts "$starts" \
  --agent "$agent" --sims "$sims" --workers "$workers" --worker "$worker" --out '{out}' "$@"
echo "$(date -u +%FT%TZ) DONE play $id" >> experiments/combat-pv/RUNBOOK.md
