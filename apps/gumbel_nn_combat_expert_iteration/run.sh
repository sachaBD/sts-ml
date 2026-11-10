#!/usr/bin/env bash
# Launch, monitor and stop runs. A run is recorded under runs/ by runs.run (runs/README.md).
#   run.sh CONFIG.toml [--scratch] [--overwrite]
#   run.sh status RUN_DIRECTORY [--json]
#   run.sh stop RUN_DIRECTORY
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
usage() { echo "usage: $0 CONFIG.toml [--scratch] [--overwrite] | status RUN [--json] | stop RUN" >&2; exit 2; }
case ${1:-} in
    stop|status)
        run=$(realpath "${2:?usage: run.sh stop|status RUN_DIRECTORY}")
        out=$run
        [[ ! -d $run/out ]] || out=$run/out
        if [[ $1 == stop ]]; then
            [[ -f $out/training/status.json ]] || { echo 'Training has not initialized yet; retry shortly.' >&2; exit 1; }
            touch "$out/training/STOP"
            echo 'Stop requested: finishing the current update and scheduled evaluation, then exiting cleanly.'
        else
            cd "$root"
            exec .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.dashboard "$out" "${@:3}"
        fi
        ;;
    ''|-*) usage ;;
    *)
        config=$(realpath "$1")  # before cd, so CONFIG is relative to where you ran this
        shift
        run_args=(--live)
        for arg in "$@"; do
            case "$arg" in --scratch | --overwrite) run_args+=("$arg") ;; *) usage ;; esac
        done
        cd "$root"
        export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
        prepare=(.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.prepare "$config")
        id=$(.venv/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run"]["id"])' "$config")
        while IFS= read -r input; do
            [[ -z $input ]] || run_args+=(--input "$input")
        done < <("${prepare[@]}" --inputs)
        exec .venv/bin/python -m runs.run gumbel_nn_combat_v1 "$id" "${run_args[@]}" -- "${prepare[@]}" --out '{out}'
        ;;
esac
