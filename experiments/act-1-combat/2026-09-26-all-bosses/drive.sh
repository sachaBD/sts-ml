#!/usr/bin/env bash
# Run experiment steps in order, from the repo root:
#   experiments/act-1-combat/2026-09-26-all-bosses/drive.sh STEP [STEP ...]
# STEP is a config name without .toml (e.g. ab-gen0, ab-teacher-dev) or "par:A+B" to run two steps concurrently.
# Before each step, every "value_net_v1/<date>/ab-*" reference in the configs is pointed at the run's real date
# (run ids carry the UTC start date). Stops at the first failure. Times go to results/drive.log.
set -uo pipefail
HERE=experiments/act-1-combat/2026-09-26-all-bosses
LOG=$HERE/results/drive.log

fix_refs() {
  for f in "$HERE"/configs/*.toml; do
    for id in $(grep -o 'value_net_v1/[0-9-]*/ab-[a-z0-9-]*' "$f" | sed 's#.*/##' | sort -u); do
      real=$(ls -d runs/schema=value_net_v1/date=*/id="$id" 2>/dev/null | tail -1)
      [ -n "$real" ] || continue
      date=$(echo "$real" | sed 's#.*date=\([^/]*\)/.*#\1#')
      sed -i "s#value_net_v1/[0-9-]*/$id\"#value_net_v1/$date/$id\"#" "$f"
    done
  done
}

app_for() {
  case "$1" in
    *-dev) echo apps/value_play/run.sh ;;
    ab-selfplay-*) echo apps/fight_resample/run.sh ;;
    *) echo apps/value_train/run.sh ;;
  esac
}

run_step() {
  local step=$1 start=$(date +%s)
  fix_refs
  echo "$(date -u +%FT%TZ) start $step" >> "$LOG"
  "$(app_for "$step")" "$HERE/configs/$step.toml" > "$HERE/results/$step.console.log" 2>&1
  local code=$?
  echo "$(date -u +%FT%TZ) end   $step exit=$code minutes=$(( ($(date +%s) - start) / 60 ))" >> "$LOG"
  return $code
}

for step in "$@"; do
  if [[ $step == par:* ]]; then
    IFS=+ read -ra pair <<< "${step#par:}"
    pids=()
    for s in "${pair[@]}"; do run_step "$s" & pids+=($!); sleep 20; done
    for p in "${pids[@]}"; do wait "$p" || { echo "FAILED in $step" >> "$LOG"; exit 1; }; done
  else
    run_step "$step" || { echo "FAILED $step" >> "$LOG"; exit 1; }
  fi
done
echo "$(date -u +%FT%TZ) all done: $*" >> "$LOG"
