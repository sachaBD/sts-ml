#!/usr/bin/env bash
# Stage 3b: collection with r0b-v1td (best Act 3 / Act 4 entry rate on dev; hybrid showed its Act-3 gain comes from
# Act 1-2 choices, not Act-3 decisions). ah-c03 (hybrid) was stopped after ~160 runs and is kept as partial data.
set -uo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
TD=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt
$P ah-c04 "$TD" 971000006000 2000 0.05 0.15 --key-rule-p 0.5
