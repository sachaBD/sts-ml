#!/usr/bin/env bash
# gen0 regularisation sweep (scratch runs, 5 epochs each), sequential.
cd "$(dirname "$0")/../../.."
for n in wd05 lr3e4 small; do
  ./apps/value_train/run.sh experiments/act-1-combat/2026-09-26-all-bosses/configs/sweep/$n.toml --scratch \
    > experiments/act-1-combat/2026-09-26-all-bosses/results/sweep-$n.console.log 2>&1
done
./apps/value_train/run.sh experiments/act-1-combat/2026-09-26-all-bosses/configs/sweep/wd05.toml --scratch --overwrite \
  > experiments/act-1-combat/2026-09-26-all-bosses/results/sweep-wd05.console.log 2>&1
