#!/usr/bin/env bash
# Stage 2b: dev gate of v3.2 (r0b, MC on ah-c01+c02, from scratch), key rule on (same crutch as v1), greedy.
set -uo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
$P ah-dev-r0b-v32 runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v32/out/model.pt 961000000000 600 0 0 --key-rule
