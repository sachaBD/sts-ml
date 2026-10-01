#!/usr/bin/env bash
# train_one.sh PREFIX TOPOLOGY LR SEED DATA.txt EVAL.txt  -> run combat_outcome_v1/<date>/PREFIX-TOPOLOGY-lrLR-sSEED (augmented arm only)
# Skips if a done run with that id exists (safe resume). Never overwrites.
set -euo pipefail
cd "$(dirname "$0")/../.."
prefix=$1 topo=$2 lr=$3 seed=$4 data=$5 eval=$6
id=$(echo "$prefix-$topo-lr$lr-s$seed" | tr A-Z a-z)
if grep -qs '"status": "done"' runs/schema=combat_outcome_v1/date=*/id=$id/run.json; then echo "SKIP $id"; exit 0; fi
export PYTHONPATH="$PWD" CARD_OUTCOME_LR=$lr CARD_OUTCOME_SEED=$seed
natural=combat_transition_v1/2026-09-29/act1-eval-mcts-a20-checked
mapfile -t inputs < <(grep -vE '^[[:space:]]*(#|$)' "$data"); mapfile -t evaluation < <(grep -vE '^[[:space:]]*(#|$)' "$eval")
lineage=(--input "$natural"); for r in "${inputs[@]}"; do lineage+=(--input "$r"); done
.venv/bin/python -m runs.run combat_outcome_v1 "$id" --note "topology=$topo lr=$lr seed=$seed pair_w=${CARD_OUTCOME_PAIR_W:-0}" "${lineage[@]}" -- \
  .venv/bin/python apps/combat_transition/train_marginals.py "$natural" --marginals "${inputs[@]}" \
  --eval-marginals "${evaluation[@]}" --epochs 60 --device cuda --out '{out}' \
  --topology "models/combat_outcome/architectures/$topo.toml" --arms augmented > "experiments/topology-sweep/logs/$id.log" 2>&1 \
  && echo "DONE $id" || { echo "FAIL $id"; exit 1; }
