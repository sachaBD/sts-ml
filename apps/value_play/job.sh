#!/usr/bin/env bash
# Internal job: build the worker in its own build dir (never build/ or build-dev/), then replay the
# fights. play.py copies the built worker into the run's out/ and plays with that copy.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
cmake -S . -B build-valexp -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-valexp --target value_play_worker --parallel >/dev/null
exec .venv/bin/python apps/value_play/play.py "$@"
