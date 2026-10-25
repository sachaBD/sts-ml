#!/usr/bin/env bash
# ./apps/az_train/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
# Needs the JAX stack: uv sync --group jax (see pyproject.toml).
exec "$(dirname "$0")/../common/launch.sh" az_champ_v1 apps/az_train/train.py -- "$@"
