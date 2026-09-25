#!/usr/bin/env bash
# Approved Stage A batch; stop on execution failure or baseline trajectory mismatch.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "Stage A started $(date -u --iso-8601=seconds)"
./apps/value_play/run.sh rundecks/slime-v7/configs/slime-v7-r00.toml
.venv/bin/python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
old = Path('runs/schema=combat_v3/date=2026-09-24/id=slime-v6-teacher-play/out')
candidates = list(Path('runs/schema=combat_v3').glob('date=*/id=slime-v7-r00/out'))
assert len(candidates) == 1, candidates
new = candidates[0]
cols = ['episode_id', 'decision_index', 'chosen_action', 'won', 'final_hp', 'potions', 'terminal_value']
def decisions(folder):
    result = {}
    for path in folder.glob('part-*.parquet'):
        for row in pq.ParquetFile(path).read(columns=['row_kind', *cols]).to_pylist():
            if row['row_kind'] == 'decision':
                key = (row['episode_id'], row['decision_index'])
                result[key] = tuple(row[c] for c in cols)
    return result
before, after = decisions(old), decisions(new)
assert before and before == after, 'STOP: v6 baseline trajectory/outcome mismatch'
print(f'Baseline reproduced: {len(after)} decisions, identical actions and outcomes.', flush=True)
PY
for arm in r10 r01 r11; do
    echo "Launching $arm $(date -u --iso-8601=seconds)"
    ./apps/value_play/run.sh "rundecks/slime-v7/configs/slime-v7-$arm.toml"
done
echo "Stage A completed $(date -u --iso-8601=seconds)"
