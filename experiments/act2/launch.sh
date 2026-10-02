#!/usr/bin/env bash
# Act 1+2 expert-iteration controller (see driver.py). Resume: rerun this script unchanged (same id).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
INIT=runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt
exec .venv/bin/python -m runs.run overworld_v1 act2-controller --no-compact \
  --input "$INIT" --input build/act2/run_rl_worker -- \
  .venv/bin/python experiments/act2/driver.py --out '{out}' --worker build/act2/run_rl_worker --init "$INIT" \
  --hours 9 --workers 10 --batch 3000 --eval-seeds 600 --fresh-seeds 800 --round-hours 2.0 --final-reserve 1.0
