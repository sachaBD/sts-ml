#!/usr/bin/env bash
# The job apps/common/launch.sh hands to the launcher: build the app's worker (if any), then run the app.
#   job.sh [--build BUILD_DIR TARGET] SCRIPT ARGS...
# Each app builds in its own BUILD_DIR, so a rebuild elsewhere can't change a running app's worker.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == --build ]]; then
    cmake -S . -B "$2" -DCMAKE_BUILD_TYPE=Release >/dev/null
    nice cmake --build "$2" --target "$3" --parallel >/dev/null
    shift 3
fi
exec .venv/bin/python "$@"
