#!/usr/bin/env bash
# Launch a config-driven bootstrap run; the generic launcher owns the run directory.
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
id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
schema=$(PYTHONPATH=python .venv/bin/python -c 'from sts_combat_rl.schemas.combat_v3 import NAME; print(NAME)')
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" exec .venv/bin/python -m sts_combat_rl.run "$schema" "$id" "${run_args[@]}" -- \
    apps/bootstrap/job.sh "$config" --out '{out}'
