#!/usr/bin/env bash
# Launch a config-driven bootstrap run; the generic launcher owns the run directory.
set -euo pipefail
cd "$(dirname "$0")/../.."

if (($# < 1 || $# > 2)) || { (($# == 2)) && [[ $2 != --scratch ]]; }; then
    echo "usage: $0 CONFIG.toml [--scratch]" >&2
    exit 2
fi
config=$(realpath "$1")
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
schema=$(PYTHONPATH=apps/bootstrap .venv/bin/python -c 'import schema; print(schema.NAME)')  # apps/bootstrap/schema.py
run_args=(--live)
if [[ ${2:-} == --scratch ]]; then run_args+=(--scratch); fi
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" exec .venv/bin/python -m sts_combat_rl.run "$schema" "$id" "${run_args[@]}" -- \
    apps/bootstrap/job.sh "$config" --out '{out}'
