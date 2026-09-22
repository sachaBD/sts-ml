#!/usr/bin/env bash
# Internal run job: build the worker if needed, then run the orchestrator.
# Called by apps/bootstrap/run.sh after the generic run launcher has supplied --out.
# Ctrl+C (or SIGTERM) stops it cleanly: fights in flight finish, shards flush, summary.json is written.
set -euo pipefail
cd "$(dirname "$0")/../.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target bootstrap_fight_worker --parallel >/dev/null

exec .venv/bin/python apps/bootstrap/generate.py "$@"
