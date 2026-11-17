#!/usr/bin/env bash
# Watch a megacrit_dump pull: progress line, recent errors, status. Usage: apps/megacrit_dump/monitor.sh [run dir] [secs]
cd "$(dirname "$0")/../.."
RUN=${1:-$(ls -d runs/schema=megacrit_runs_v1/date=*/id=ironclad-a20-champ-10k | tail -1)}
EVERY=${2:-10}
while true; do
  clear
  echo "run:    $RUN"
  echo "status: $(grep -o '"status": "[a-z]*"' "$RUN/run.json" | cut -d'"' -f4)   ($(date +%T))"
  echo "latest: $(grep 'files,' "$RUN/logs/stdout.log" | tail -1)"
  echo "--- stderr (last 5) ---"; tail -5 "$RUN/logs/stderr.log"
  if [ -f "$RUN/out/summary.json" ]; then
    echo "--- finished ---"
    grep -E '"(files_processed|kept_champ_wins|wall_seconds|stop_reason|throttled|files_failed_this_invocation)"' "$RUN/out/summary.json"
    break
  fi
  sleep "$EVERY"
done
