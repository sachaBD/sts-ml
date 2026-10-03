#!/usr/bin/env bash
# Round 0a: offline topology screen on ah-c01 only (MC targets, lam 1). Runs while ah-c02 collects.
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/act3-heart/train.sh
C01=runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out
INC=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt
A=agents/overworld/value/architectures
$T ah-r0a-v1init --init "$INC" --data $C01 --lam 1 --epochs 20
$T ah-r0a-v1 --data $C01 --lam 1 --epochs 20
$T ah-r0a-v31 --arch-spec $A/rp3.1-w64-h128-l2.toml --data $C01 --lam 1 --epochs 20
$T ah-r0a-v32 --arch-spec $A/rp3.2-w64-h128-l2.toml --data $C01 --lam 1 --epochs 20
