#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
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
    *)
        config=$(realpath "${1:?usage: run.sh CONFIG.toml [--scratch] [--overwrite] | stop|status RUN_DIRECTORY}")
        cd "$root"  # the config's paths are relative to the repo root
        exec .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.launch "$config" "${@:2}"
        ;;
esac
