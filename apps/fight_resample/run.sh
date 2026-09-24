#!/usr/bin/env bash
# Launch a fight_resample run (slop_docs/apps/fight_resample.md); inputs = the source runs (+ the value run).
set -euo pipefail

usage() { echo "usage: $0 CONFIG.toml [--scratch] [--overwrite]" >&2; exit 2; }
(($# >= 1 && $# <= 3)) || usage
config=$(realpath "$1")  # before cd, so CONFIG is relative to where you ran this
shift
run_args=(--live)
for arg in "$@"; do
    case "$arg" in
        --scratch | --overwrite) run_args+=("$arg") ;;
        *) usage ;;
    esac
done
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
schema=$(.venv/bin/python -c 'from sts_combat_rl.schemas.combat_v3 import NAME; print(NAME)')
mapfile -t inputs < <(.venv/bin/python apps/fight_resample/generate.py "$config" --inputs)
for input in "${inputs[@]}"; do run_args+=(--input "$input"); done
exec .venv/bin/python -m sts_combat_rl.run "$schema" "$id" "${run_args[@]}" -- apps/fight_resample/job.sh "$config" --out '{out}'
