#!/usr/bin/env bash
# Act 2 boss fights: 20k vs 100k guided-rollout MCTS on the same 300 rebuilt fights (100 per boss).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
SRC=runs/schema=overworld_v1/date=2026-10-02/id=act2-fresh-final/out
BIN=/tmp/boss_rebuild_worker.$$; cp build/act2/boss_rebuild_worker "$BIN"
exec .venv/bin/python -m runs.run boss_bench_v1 act2-boss-20k-vs-100k --no-compact --input "$SRC" -- \
  .venv/bin/python apps/boss_rebuild/bench.py --source "$SRC" --out '{out}' --worker "$BIN" \
  --per-boss 100 --sims 20000 100000 --workers 10
