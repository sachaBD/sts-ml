#!/usr/bin/env bash
# ./apps/fight_resample/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" combat_v3 apps/fight_resample/generate.py build/resample fight_resample_worker -- "$@"
