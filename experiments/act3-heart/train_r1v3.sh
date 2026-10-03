#!/usr/bin/env bash
# Round 1 v3: continue r0d-v32d with dense distillation from ah-r1-v1td + MC, all Heart-mode collections (decay .85).
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/act3-heart/train.sh
D="runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c03/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c04/out"
$T ah-r1-v32d --init runs/schema=value_net_v1/date=2026-10-03/id=ah-r0d-v32d/out/model.pt --data $D --lam 1 --decay .85 --epochs 20 \
  --distill runs/schema=value_net_v1/date=2026-10-03/id=ah-r1-v1td/out/model.pt --distill-weight 2 --batch-size 128
