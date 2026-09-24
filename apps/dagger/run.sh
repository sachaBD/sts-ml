#!/usr/bin/env bash
# ./apps/dagger/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" combat_v3 apps/dagger/generate.py build/dagger dagger_worker -- "$@"
