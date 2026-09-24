#!/usr/bin/env bash
# ./apps/compare_fights/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" fight_comparison_v1 apps/compare_fights/compare.py -- "$@"
