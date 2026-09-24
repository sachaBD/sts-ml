#!/usr/bin/env bash
# Internal job: build the worker in its own build dir (never build/ or build-dev/), then collect.
# generate.py copies the built worker and the learner weights into the run's out/ and plays with those copies.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
cmake -S . -B build-dagger -DCMAKE_BUILD_TYPE=Release >/dev/null
nice cmake --build build-dagger --target dagger_worker --parallel 2 >/dev/null
exec .venv/bin/python apps/dagger/generate.py "$@"
