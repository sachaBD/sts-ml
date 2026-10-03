#!/usr/bin/env bash
# Round 0b: candidates on ah-c01 + ah-c02 (~4000 runs). v1: MC from the Act-2 incumbent, and TD(.7) continuing
# r0a-v1init. v3.1 / v3.2 from scratch (MC), trained in parallel on the GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/act3-heart/train.sh
D="runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out $(ls -d runs/schema=overworld_v1/date=*/id=ah-c02/out)"
INC=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt
R0A=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0a-v1init/out/model.pt
A=agents/overworld/value/architectures
$T ah-r0b-v31 --arch-spec $A/rp3.1-w64-h128-l2.toml --data $D --lam 1 --epochs 20 &
$T ah-r0b-v32 --arch-spec $A/rp3.2-w64-h128-l2.toml --data $D --lam 1 --epochs 20 &
$T ah-r0b-v1init --init "$INC" --data $D --lam 1 --epochs 20
$T ah-r0b-v1td --init "$R0A" --data $D --lam .7 --epochs 20
wait
