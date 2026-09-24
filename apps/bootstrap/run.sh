#!/usr/bin/env bash
# ./apps/bootstrap/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" combat_v3 apps/bootstrap/generate.py build/main bootstrap_fight_worker -- "$@"
