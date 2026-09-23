#!/usr/bin/env bash
# Internal job: train a value network from the config into the managed out dir.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python apps/value_train/train.py "$@"
