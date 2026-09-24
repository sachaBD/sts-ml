#!/usr/bin/env bash
# Launch a compare_fights run (slop_docs/apps/compare_fights.md); inputs = [baseline, candidate].
set -euo pipefail

usage() { echo "usage: $0 CONFIG.toml [--scratch] [--overwrite]" >&2; exit 2; }
(($# >= 1 && $# <= 3)) || usage
config=$(realpath "$1")  # before cd, so CONFIG is relative to where you ran this
shift
extra=()
for arg in "$@"; do
    case "$arg" in
        --scratch | --overwrite) extra+=("$arg") ;;
        *) usage ;;
    esac
done
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
schema=$(PYTHONPATH="apps/compare_fights:$PYTHONPATH" .venv/bin/python -c 'import compare; print(compare.NAME)')
mapfile -t inputs < <(.venv/bin/python apps/compare_fights/compare.py "$config" --inputs)
run_args=(--live)
for input in "${inputs[@]}"; do run_args+=(--input "$input"); done
run_args+=("${extra[@]}")
exec .venv/bin/python -m sts_combat_rl.run "$schema" "$id" "${run_args[@]}" -- apps/compare_fights/job.sh "$config" --out '{out}'
