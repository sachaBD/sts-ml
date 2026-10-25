#!/usr/bin/env bash
# Gumbel v2 (equal compute per update), then test vs stored ten-deck PUCT2k (174/300) and MCTS20k (124/300). ~1 h.
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=experiments/25-10-26/a-gumbel
./apps/combat_expert_iteration/run.sh $D/config/gumbel_panache_v2.toml > $D/logs/gumbel_v2.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -u $D/test_panache.py --workers 10
