#!/usr/bin/env bash
# One managed overworld value training job (runs.run, schema value_net_v1).
#   train.sh NAME (--init CKPT | --arch-spec TOML | --arch JSON) --data DIR... [more train.py args]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2
NAME=$1; shift
echo "$(date -u +%FT%TZ) START train $NAME $*" >> experiments/act3-heart/RUNBOOK.md
.venv/bin/python -m runs.run value_net_v1 "$NAME" --no-compact -- \
  .venv/bin/python apps/run_rl/train.py --out '{out}/model.pt' --target full "$@"
echo "$(date -u +%FT%TZ) DONE train $NAME" >> experiments/act3-heart/RUNBOOK.md
