#!/usr/bin/env bash
# ./apps/combat_expert_iteration/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
# Uses the frozen pv_worker of [run].worker_run, so no build step.
exec "$(dirname "$0")/../common/launch.sh" combat_v4 apps/combat_expert_iteration/expert_iteration.py -- "$@"
