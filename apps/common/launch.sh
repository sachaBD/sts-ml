#!/usr/bin/env bash
# Launch an app run through runs.run, which owns the run directory (runs/README.md).
# Each apps/<app>/run.sh is a one-line call to this:
#   launch.sh SCHEMA SCRIPT CONFIG.toml [--scratch] [--overwrite]
# `SCRIPT CONFIG --inputs` prints the run's input run ids (recorded as lineage); the job is SCRIPT CONFIG --out <run out dir>.
set -euo pipefail

schema=$1 script=$2
shift 2
usage() { echo "usage: $(dirname "$script")/run.sh CONFIG.toml [--scratch] [--overwrite]" >&2; exit 2; }
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
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
inputs=$(.venv/bin/python "$script" "$config" --inputs)
while IFS= read -r input; do
    if [[ -n $input ]]; then run_args+=(--input "$input"); fi
done <<< "$inputs"
exec .venv/bin/python -m runs.run "$schema" "$id" "${run_args[@]}" -- .venv/bin/python "$script" "$config" --out '{out}'
