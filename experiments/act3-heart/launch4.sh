#!/usr/bin/env bash
# Stage 3: hybrid policy = Act-2 incumbent (acts 1-2) + r0b-v1td (full-target, from Act 3) — dev gate, then
# collection ah-c03 with the hybrid (more Act 3 data). Key rule on (dev) / on half the runs (collection).
set -uo pipefail
cd "$(dirname "$0")/../.."
P=experiments/act3-heart/play.sh
INC=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt
LATE=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt
$P ah-dev-hyb "$INC" 961000000000 600 0 0 --key-rule --late-ckpt "$LATE" --late-act 3
$P ah-c03 "$INC" 971000004000 2000 0.05 0.15 --key-rule-p 0.5 --late-ckpt "$LATE" --late-act 3
