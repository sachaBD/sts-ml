#!/usr/bin/env bash
# Launch a value_play run (slop_docs/apps/value_play.md); inputs = [value run, its data runs].
set -euo pipefail

if (($# < 1 || $# > 3)); then
    echo "usage: $0 CONFIG.toml [--scratch] [--overwrite]" >&2
    exit 2
fi
config=$(realpath "$1")  # before cd, so CONFIG is relative to where you ran this
shift
scratch=false
overwrite=false
for arg in "$@"; do
    case "$arg" in
        --scratch) scratch=true ;;
        --overwrite) overwrite=true ;;
        *) echo "usage: $0 CONFIG.toml [--scratch] [--overwrite]" >&2; exit 2 ;;
    esac
done
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
schema=$(.venv/bin/python -c 'from sts_combat_rl.schemas.combat_v3 import NAME; print(NAME)')
mapfile -t inputs < <(.venv/bin/python apps/value_play/play.py "$config" --inputs)
run_args=(--live)
for input in "${inputs[@]}"; do run_args+=(--input "$input"); done
if $scratch; then run_args+=(--scratch); fi
if $overwrite; then run_args+=(--overwrite); fi
exec .venv/bin/python -m sts_combat_rl.run "$schema" "$id" "${run_args[@]}" -- apps/value_play/job.sh "$config" --out '{out}'
