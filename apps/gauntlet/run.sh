#!/usr/bin/env bash
# ./apps/gauntlet/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" gauntlet_v1 apps/gauntlet/gauntlet.py build/main gauntlet_worker -- "$@"
