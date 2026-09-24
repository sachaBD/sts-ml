#!/usr/bin/env bash
# Internal job: build the worker in its own build dir, then resample the fights.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
cmake -S . -B build-resample -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-resample --target fight_resample_worker --parallel >/dev/null
exec .venv/bin/python apps/fight_resample/generate.py "$@"
