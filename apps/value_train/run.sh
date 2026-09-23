#!/usr/bin/env bash
# Launch a config-driven value-network training run.
set -euo pipefail

if (($# < 1 || $# > 2)) || { (($# == 2)) && [[ $2 != --scratch ]]; }; then
    echo "usage: $0 CONFIG.toml [--scratch]" >&2
    exit 2
fi

config=$(realpath "$1")
cd "$(dirname "$0")/../.."
mapfile -t fields < <(.venv/bin/python - <<'PY' "$config"
import sys, tomllib
cfg = tomllib.load(open(sys.argv[1], "rb"))
run = cfg["run"]
print(run["id"])
inputs = run.get("inputs", run.get("input"))
if inputs is None:
    inputs = cfg.get("data", {}).get("paths", [])
if isinstance(inputs, str):
    inputs = [inputs]
for item in inputs:
    print(item)
PY
)

id=${fields[0]}
run_args=(--live)
if [[ ${2:-} == --scratch ]]; then run_args+=(--scratch); fi
for input in "${fields[@]:1}"; do
    run_args+=(--input "$input")
done

PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" exec .venv/bin/python -m sts_combat_rl.run value_net_v1 "$id" "${run_args[@]}" -- \
    apps/value_train/job.sh "$config" --out '{out}'
