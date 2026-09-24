#!/usr/bin/env bash
# Launch a compare_fights run (slop_docs/apps/compare_fights.md); inputs = [baseline, candidate].
set -euo pipefail

if (($# < 1 || $# > 2)) || { (($# == 2)) && [[ $2 != --scratch ]]; }; then
    echo "usage: $0 CONFIG.toml [--scratch]" >&2
    exit 2
fi
config=$(realpath "$1")  # before cd, so CONFIG is relative to where you ran this
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
schema=$(PYTHONPATH="apps/compare_fights:$PYTHONPATH" .venv/bin/python -c 'import compare; print(compare.NAME)')
mapfile -t inputs < <(.venv/bin/python apps/compare_fights/compare.py "$config" --inputs)
run_args=(--live)
for input in "${inputs[@]}"; do run_args+=(--input "$input"); done
if [[ ${2:-} == --scratch ]]; then run_args+=(--scratch); fi
exec .venv/bin/python -m sts_combat_rl.run "$schema" "$id" "${run_args[@]}" -- apps/compare_fights/job.sh "$config" --out '{out}'
