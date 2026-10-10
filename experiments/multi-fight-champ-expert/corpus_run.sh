#!/usr/bin/env bash
# Fixed local pilot. No Azure; no overwrite. Existing journals resume only identical jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/schema=combat_v4/date=2026-10-09/id=champ-corpus-pilot-v1
exec .venv/bin/python -u -m apps.combat_expert_iteration.corpus --run "$RUN" --workers 10 "$@"
