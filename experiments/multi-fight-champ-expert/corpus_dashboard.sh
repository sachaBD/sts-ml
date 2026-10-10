#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="${1:-runs/schema=combat_v4/date=2026-10-09/id=champ-corpus-pilot-v1}"
exec .venv/bin/python -m apps.combat_expert_iteration.corpus_status --run "$RUN"
