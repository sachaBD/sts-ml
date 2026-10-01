#!/usr/bin/env bash
# Stage 1: v4 loop, 2 iterations (gen -> train -> eval), every decision by V. ~80 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=runs/schema=run_rl_v1/date=2026-10-01/id=v4-all
V2=runs/schema=run_rl_v1/date=2026-10-01/id=v2-rest-path/out
V3=runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out
mkdir -p $R/out $R/logs
PYTHONPATH=. exec .venv/bin/python apps/run_rl/loop.py --root $R/out --iters 2 --window 8 --decay 0.85 \
  --decide rest path shop event neow --init $V3/iter003/model.pt \
  --extra-data $V2/iter002/data $V2/iter003/data $V2/iter004/data $V2/iter005/data \
               $V3/iter000/data $V3/iter001/data $V3/iter002/data $V3/iter003/data \
  --seed-offset 40000000 --worker experiments/run-rl-v4/bin/run_rl_worker >> $R/logs/stdout.log 2>&1
