#!/usr/bin/env bash
# Stage 5: fresh-seed final (800 seeds 981e9+, never used for training/selection): selected r0d-v32d (distilled v3.2)
# vs the Act-2 incumbent; then dev gate of ah-r1-v32d (round-1 v3.2d); then fresh r0b-v1td (third arm).
set -uo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
V=runs/schema=value_net_v1/date=2026-10-03
INC=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt
$P ah-fresh-r0d-v32d $V/id=ah-r0d-v32d/out/model.pt 981000000000 800 0 0 --key-rule
$P ah-fresh-inc "$INC" 981000000000 800 0 0 --key-rule
$P ah-dev-r1-v32d $V/id=ah-r1-v32d/out/model.pt 961000000000 600 0 0 --key-rule
$P ah-fresh-r0b-v1td $V/id=ah-r0b-v1td/out/model.pt 981000000000 800 0 0 --key-rule
