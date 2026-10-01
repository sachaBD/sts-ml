#!/usr/bin/env bash
# Train/evaluate through the existing run launcher. EVAL.txt stays fixed across stages.
# train.sh MODEL_ID CUMULATIVE_DATA.txt EVAL.txt [EPOCHS]
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# -ge 3 && $# -le 4 ]] || { echo 'usage: train.sh MODEL_ID CUMULATIVE_DATA.txt EVAL.txt [EPOCHS]' >&2; exit 2; }
id=$1; data=$2; eval=$3; epochs=${4:-60}
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
natural=combat_transition_v1/2026-09-29/act1-eval-mcts-a20-checked
mapfile -t inputs < <(grep -vE '^[[:space:]]*(#|$)' "$data")
mapfile -t evaluation < <(grep -vE '^[[:space:]]*(#|$)' "$eval")
[[ ${#inputs[@]} -gt 0 && ${#evaluation[@]} -gt 0 ]] || { echo 'Empty input/evaluation manifest' >&2; exit 2; }
lineage=(--input "$natural")
for run in "${inputs[@]}"; do lineage+=(--input "$run"); done
exec .venv/bin/python -m sts_combat_rl.run combat_outcome_v1 "$id" "${lineage[@]}" -- \
  .venv/bin/python apps/combat_transition/train_marginals.py "$natural" \
  --marginals "${inputs[@]}" --eval-marginals "${evaluation[@]}" \
  --epochs "$epochs" --device cuda --out '{out}'
