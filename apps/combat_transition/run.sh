#!/usr/bin/env bash
# ./apps/combat_transition/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" combat_transition_v1 apps/combat_transition/extract.py build/main combat_transition_worker -- "$@"
