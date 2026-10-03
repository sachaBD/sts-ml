#!/usr/bin/env bash
# Stage 4b: dev gate of ah-r1-v1td (round 1 v1), after launch6.
set -uo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f "bash experiments/act3-heart/launch6.sh" >/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) LEAD LAUNCH (chained) ah-dev-r1-v1td" >> experiments/act3-heart/RUNBOOK.md
P=experiments/act3-heart/play.sh
$P ah-dev-r1-v1td runs/schema=value_net_v1/date=2026-10-03/id=ah-r1-v1td/out/model.pt 961000000000 600 0 0 --key-rule
