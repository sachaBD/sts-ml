#!/usr/bin/env bash
# Unattended ~7.5 h job: train the five-fight network, then test it. See RUNBOOK.md.
#   1. apps/combat_expert_iteration/run.sh  config/champ_five_rollout.toml          (~6-6.5 h)
#   2. apps/combat_expert_iteration/evaluate.sh  config/test_champ_five.template.toml (filled in)  (~65 min)
# Stops at the first failure. Never retries, never overwrites a run.
# Log: scratch/multi-fight-champ-expert/overnight-<UTC time>.log (path printed at start).
set -euo pipefail
cd "$(dirname "$0")/../.."
APP=apps/combat_expert_iteration
LOGDIR=scratch/multi-fight-champ-expert
mkdir -p "$LOGDIR"
LOG="$LOGDIR/overnight-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "log: $PWD/$LOG"
exec >>"$LOG" 2>&1
say() { echo "$(date -u +%FT%TZ) overnight: $*"; }
trap 'say "FAILED (exit $?) at line $LINENO; see the run logs above"' ERR

say "preflight"
if pgrep -f "pv_worker (play|teacher)" >/dev/null; then say "other pv_worker games are running; refusing to start"; exit 1; fi
for id in champ-five-rollout-v1 champ-five-test-v1; do
    if compgen -G "runs/schema=combat_v4/date=*/id=$id" >/dev/null; then say "run $id already exists; refusing to overwrite"; exit 1; fi
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
df -h . | tail -1

say "1/2 training: $APP/config/champ_five_rollout.toml"
"$APP/run.sh" "$APP/config/champ_five_rollout.toml"
TRAIN=$(ls -d runs/schema=combat_v4/date=*/id=champ-five-rollout-v1)
test -f "$TRAIN/out/update010/model/model.onnx"
say "training done: $TRAIN"

TEST_CONFIG="$LOGDIR/test_champ_five.toml"
sed "s|__TRAIN_OUT__|$TRAIN/out|" "$APP/config/test_champ_five.template.toml" > "$TEST_CONFIG"
say "2/2 test: $TEST_CONFIG"
"$APP/evaluate.sh" "$TEST_CONFIG"
say "DONE: $(ls -d runs/schema=combat_v4/date=*/id=champ-five-test-v1)/out/REPORT.md"
