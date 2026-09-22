#!/usr/bin/env bash
# User-facing bootstrap launcher.
#
#   ./apps/bootstrap/run.sh slime-bootstrap --workers 12 --forever
#
# The generic run launcher creates runs/run_id=... and supplies --out privately.
set -euo pipefail
cd "$(dirname "$0")/../.."

if (($# == 0)) || [[ $1 == -* ]]; then
    echo "usage: $0 RUN_NAME [bootstrap generator options]" >&2
    echo "example: $0 slime-bootstrap --workers 12 --forever" >&2
    exit 2
fi

name=$1
shift
run_args=()
if [[ ${1:-} == "--scratch" ]]; then
    run_args+=(--scratch)
    shift
fi
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" exec .venv/bin/python -m sts_combat_rl.run gen "$name" "${run_args[@]}" -- \
    apps/bootstrap/job.sh --out '{out}' "$@"
