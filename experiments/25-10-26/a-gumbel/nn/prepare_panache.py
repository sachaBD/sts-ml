"""Extract stored panache starts and MCTS bootstrap games. No gameplay.

  .venv/bin/python experiments/25-10-26/a-gumbel/nn/prepare_panache.py OUTPUT_DIR
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
source = Path('runs/schema=combat_v4/date=2026-10-09')
starts = json.loads((source / 'id=champ-ten-test-u6-v1/out/starts.json').read_text())['panache']
(out / 'panache_start.json').write_text(json.dumps(dict(fight_id='panache', start=starts[0]['start'])))
with (out / 'panache_test.jsonl').open('w') as target:
    for row in starts:
        target.write(json.dumps(dict(fight_id=row['fight_id'], start=row['start'])) + '\n')
with (source / 'id=champ-ten-rollout-v1/out/bootstrap/play/results.jsonl').open() as games:
    with (out / 'panache_mcts_bootstrap.jsonl').open('w') as target:
        for line in games:
            if ':panache:' in json.loads(line)['fight_id']:
                target.write(line)
