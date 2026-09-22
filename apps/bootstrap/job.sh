#!/usr/bin/env bash
# Internal job: build C++ once, then generate fights from the config.
set -euo pipefail
cd "$(dirname "$0")/../.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target bootstrap_fight_worker --parallel >/dev/null
exec .venv/bin/python apps/bootstrap/generate.py "$@"
