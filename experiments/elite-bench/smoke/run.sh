#!/usr/bin/env bash
# Mini end-to-end rehearsal of the deck in scratch/ (12 fights).
set -e
cd "$(dirname "$0")/../../.."
S=experiments/elite-bench/smoke
for r in oracle-20k oracle-50k mcts-20k-s0 mcts-20k-s1 v3-20k-s0 v3-20k-s1 mcts-1k-s0; do
  ./apps/value_play/run.sh $S/configs/bsmoke-$r.toml --scratch > $S/$r.log 2>&1 || { echo "FAIL $r"; tail $S/$r.log; exit 1; }
done
PYTHONPATH=. .venv/bin/python experiments/elite-bench/pivotal.py --prefix bsmoke --store scratch --out $S/pivotal.csv
for r in mcts-20k-s4 v3-20k-s4 mcts-20k-p32-s0 mcts-5k-s0 v3-5k-s0; do
  ./apps/value_play/run.sh $S/configs/bsmoke-$r.toml --scratch > $S/$r.log 2>&1 || { echo "FAIL $r"; tail $S/$r.log; exit 1; }
done
PYTHONPATH=. .venv/bin/python experiments/elite-bench/analyze.py --prefix bsmoke --store scratch --output $S/analysis.md
