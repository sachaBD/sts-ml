#!/usr/bin/env bash
# Stage 2a: dev gate of the full-target v1 candidate r0b-v1td (TD .7 from r0a-v1init on ah-c01+c02), rule on, greedy.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
$P ah-dev-r0b-v1td runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt 961000000000 600 0 0 --key-rule
