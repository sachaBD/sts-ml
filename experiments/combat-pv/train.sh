#!/usr/bin/env bash
# train.sh ID DATA... -- [train.py args]   → runs/schema=pv_model_v1/date=<today>/id=ID/out (log: ../train.log)
set -euo pipefail
cd "$(dirname "$0")/../.."
id=$1; shift
data=(); while [[ $# -gt 0 && $1 != -- ]]; do data+=("$1"); shift; done; shift || true
out=runs/schema=pv_model_v1/date=$(date -u +%F)/id=$id
mkdir -p "$out/out"
echo "$(date -u +%FT%TZ) START train $id data=${data[*]} args=$*" >> experiments/combat-pv/RUNBOOK.md
.venv/bin/python -m agents.combat.pv.train --data "${data[@]}" --out "$out/out" --device cuda "$@" > "$out/train.log" 2>&1
echo "$(date -u +%FT%TZ) DONE train $id" >> experiments/combat-pv/RUNBOOK.md
