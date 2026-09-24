#!/usr/bin/env bash
# ./apps/value_train/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" value_net_v1 apps/value_train/train.py -- "$@"
