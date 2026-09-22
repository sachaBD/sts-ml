#!/usr/bin/env bash
# Bootstrap data generation: build the worker if needed, then run the orchestrator.
# Everything after this script's own flags is passed through to apps/bootstrap/generate.py.
#
#   PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run gen slime-bootstrap -- \
#       apps/bootstrap/run.sh --out {out} --workers 12 --forever
#
# Ctrl+C (or SIGTERM) stops it cleanly: fights in flight finish, shards flush, summary.json is written.
set -euo pipefail
cd "$(dirname "$0")/../.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target bootstrap_fight_worker --parallel >/dev/null

exec .venv/bin/python apps/bootstrap/generate.py "$@"
