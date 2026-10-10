#!/usr/bin/env bash
# ./apps/combat_expert_iteration/evaluate.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
# Head-to-head agent test on fresh starts (evaluate.py); uses the frozen pv_worker of [run].worker_run.
exec "$(dirname "$0")/../common/launch.sh" combat_v4 apps/combat_expert_iteration/evaluate.py -- "$@"
