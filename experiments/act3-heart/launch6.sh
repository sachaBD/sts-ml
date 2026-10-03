#!/usr/bin/env bash
# Stage 4a: dev gate of v3.2 distilled from r0b-v1td (r0d-v32d: scratch + 2x dense distillation + MC on c01+c02).
set -uo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
$P ah-dev-r0d-v32d runs/schema=value_net_v1/date=2026-10-03/id=ah-r0d-v32d/out/model.pt 961000000000 600 0 0 --key-rule
