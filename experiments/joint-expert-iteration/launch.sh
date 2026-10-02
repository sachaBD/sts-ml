#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -m runs.run overworld_v1 joint-1001-controller --no-compact \
  --input value_net_v1/2026-09-27/ab-gen1 \
  --input runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt \
  --input build/run_rl_dev/run_rl_worker -- \
  .venv/bin/python experiments/joint-expert-iteration/driver.py --out '{out}' \
  --worker build/run_rl_dev/run_rl_worker --hours 9 --workers 10 --batch 1500 --eval-seeds 400
