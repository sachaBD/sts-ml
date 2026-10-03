#!/usr/bin/env bash
# Round 1: v1 TD(.7) continuing r0b-v1td on all Heart-mode collections (c01, c02, c03 partial, c04; ~6.2k runs),
# older collections down-weighted (decay .85 per dir, as in act2).
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/act3-heart/train.sh
D="runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c03/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c04/out"
TD=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt
$T ah-r1-v1td --init $TD --data $D --lam .7 --decay .85 --epochs 20
