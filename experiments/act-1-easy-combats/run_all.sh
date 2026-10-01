#!/usr/bin/env bash
# ./experiments/act-1-easy-combats/run_all.sh NAME...   runs configs/NAME.toml one after another (from the repo root)
set -euo pipefail
for name in "$@"; do ./apps/value_play/run.sh "experiments/act-1-easy-combats/configs/$name.toml" | tail -12; done
