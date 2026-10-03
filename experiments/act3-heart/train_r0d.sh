#!/usr/bin/env bash
# Round 0d: as 0c but early stopping on MC + 2 x distillation val loss (0c stopped after epoch 1 on MC val alone), batch 128.
# on ah-c01 + ah-c02. Tests whether the scratch-v3 failure (bad relative option values) is fixable by a teacher.
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/act3-heart/train.sh
D="runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out"
TEACH=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt
A=agents/overworld/value/architectures
$T ah-r0d-v31d --arch-spec $A/rp3.1-w64-h128-l2.toml --data $D --lam 1 --epochs 20 --distill $TEACH --distill-weight 2 --batch-size 128 &
$T ah-r0d-v32d --arch-spec $A/rp3.2-w64-h128-l2.toml --data $D --lam 1 --epochs 20 --distill $TEACH --distill-weight 2 --batch-size 128 &
wait
