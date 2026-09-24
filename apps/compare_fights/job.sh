#!/usr/bin/env bash
# Internal job: compare the two runs into the managed out dir.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python apps/compare_fights/compare.py "$@"
