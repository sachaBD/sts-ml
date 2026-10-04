#!/usr/bin/env bash
# c sweep, min-max normalized Q (worker q1), boot model, 2000 sims, bench nodome
cd "$(dirname "$0")/../.."
for c in 0.5 1.25 3; do
  experiments/combat-pv/play.sh champ-bench-boot-q1-c$c runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet build/frozen/pv_worker.q1 pv 2000 9 --model runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx --c $c
done
