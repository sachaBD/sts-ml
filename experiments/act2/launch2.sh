#!/usr/bin/env bash
# Continuation controller (one more round from act2-r02-train; lower exploration per round-0 analysis).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=runs/schema=overworld_v1/date=2026-10-02
INIT=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt
exec .venv/bin/python -m runs.run overworld_v1 act2b-controller --no-compact \
  --input "$INIT" --input build/act2/run_rl_worker -- \
  .venv/bin/python experiments/act2/driver.py --out '{out}' --worker build/act2/run_rl_worker --init "$INIT" \
  --tag act2b --first-round 3 --init-trained --prior-data $R/id=act2-r01-collect/out $R/id=act2-r02-collect/out \
  --fresh-initial $R/id=act2-fresh-final/out \
  --hours 2.6 --round-hours 2.4 --final-reserve 0 --workers 10 --batch 3000 --eval-seeds 600 --fresh-seeds 800 \
  --eps 0.05 --route-p 0.15
