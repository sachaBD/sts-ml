#!/usr/bin/env bash
# Stage 2: v4 final model on the 1,000 fresh seeds of event-h0-test (830000000000+), every decision by V. ~13 min.
# Pairs with event-h0-test/out/no-event (v3, SimpleAgent events + Neow) and event-neow (v3, everything by V).
set -euo pipefail
cd "$(dirname "$0")/../.."
R=runs/schema=run_rl_v1/date=2026-10-01/id=v4-all/out
PYTHONPATH=python exec .venv/bin/python apps/run_rl/play.py --out $R/fresh-iter001 --first-seed 830000000000 --seeds 1000 \
  --policy net --ckpt $R/iter001/model.pt --decide rest path shop event neow --worker rundecks/run-rl-v4/bin/run_rl_worker \
  >> runs/schema=run_rl_v1/date=2026-10-01/id=v4-all/logs/fresh.log 2>&1
