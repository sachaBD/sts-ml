#!/usr/bin/env bash
# ./apps/value_play/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" combat_v3 apps/value_play/play.py build/valexp value_play_worker -- "$@"
