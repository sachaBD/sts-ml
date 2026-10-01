#!/usr/bin/env bash
# sweep.sh PREFIX GRID.txt DATA.txt EVAL.txt PARALLEL   GRID lines: "TOPOLOGY LR SEED". Continues past single failures.
set -uo pipefail
cd "$(dirname "$0")/../.."
prefix=$1 grid=$2 data=$3 eval=$4 par=${5:-2}
grep -vE '^[[:space:]]*(#|$)' "$grid" | xargs -P "$par" -L 1 bash -c 'experiments/topology-sweep/train_one.sh '"$prefix"' "$0" "$1" "$2" '"$data $eval"
echo "SWEEP FINISHED $(date)"
