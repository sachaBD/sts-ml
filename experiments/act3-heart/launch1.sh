#!/usr/bin/env bash
# Stage 1: two 2000-run collections with the Act-2 incumbent (v1 + key rule), then its 600-seed dev baseline.
# Seeds: collect 971e9+, dev 961e9+, fresh 981e9+ (never used for training/selection).
set -euo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
INC=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt
$P ah-c01 "$INC" 971000000000 2000 0.05 0.15 --key-rule
$P ah-c02 "$INC" 971000002000 2000 0.05 0.15 --key-rule
$P ah-dev-inc "$INC" 961000000000 600 0 0 --key-rule
