#!/usr/bin/env bash
# ./apps/card_marginals/run.sh CONFIG.toml [--scratch] [--overwrite]   (apps/common/launch.sh)
exec "$(dirname "$0")/../common/launch.sh" card_marginals_v1 apps/card_marginals/card_marginals.py build/main card_marginals_worker -- "$@"
