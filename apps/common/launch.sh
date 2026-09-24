#!/usr/bin/env bash
# Launch an app run through sts_combat_rl.run, which owns the run directory (runs/README.md).
# Each apps/<app>/run.sh is a one-line call to this:
#   launch.sh SCHEMA SCRIPT [BUILD_DIR TARGET] -- CONFIG.toml [--scratch] [--overwrite]
# `SCRIPT CONFIG --inputs` prints the run's input run ids (recorded as lineage); the job is
# apps/common/job.sh, which builds TARGET in BUILD_DIR (if given) and runs SCRIPT CONFIG --out <run out dir>.
set -euo pipefail

schema=$1 script=$2
shift 2
build=()
if [[ ${1:-} != -- ]]; then
    build=(--build "$1" "$2")
    shift 2
fi
[[ ${1:-} == -- ]] || { echo "launch.sh: expected -- before the run arguments" >&2; exit 2; }
shift

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
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
inputs=$(.venv/bin/python "$script" "$config" --inputs)
while IFS= read -r input; do
    if [[ -n $input ]]; then run_args+=(--input "$input"); fi
done <<< "$inputs"
exec .venv/bin/python -m sts_combat_rl.run "$schema" "$id" "${run_args[@]}" -- \
    apps/common/job.sh "${build[@]}" "$script" "$config" --out '{out}'
