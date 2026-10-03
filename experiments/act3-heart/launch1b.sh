#!/usr/bin/env bash
# Stage 1b (ah-c01 failed on 2 combat_v4 replay mismatches; worker now drops such records). Emerald routing now
# Act 3 only. ah-c02: key rule on half the runs (by seed) for key-variation data; dev baseline: rule on.
# Seeds: collect 971e9+, dev 961e9+, fresh 981e9+ (never used for training/selection).
set -euo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
INC=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt
$P ah-c02 "$INC" 971000002000 2000 0.05 0.15 --key-rule-p 0.5
$P ah-dev-inc "$INC" 961000000000 600 0 0 --key-rule
