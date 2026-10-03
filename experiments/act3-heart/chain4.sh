#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "experiments/act3-heart/launch3.sh" >/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) LEAD LAUNCH launch4.sh (ah-dev-hyb ~45 min, then ah-c03 ~90 min)" >> experiments/act3-heart/RUNBOOK.md
exec experiments/act3-heart/launch4.sh > experiments/act3-heart/launch4.log 2>&1
