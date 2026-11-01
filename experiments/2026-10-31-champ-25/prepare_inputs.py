"""Freeze 25 training starts and 50 validation fights/deck from the existing store. No gameplay.
Run from this worktree: python -m experiments.2026-10-31-champ-25.prepare_inputs MAIN_CHECKOUT
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

from apps.common.app import read_jsonl, write_json
from apps.gumbel_nn_combat_expert_iteration.reporting import wilson
from environments.combat.starts import fights


def family(start):
    cards = sorted(Counter((c['id'], c['upgraded'], c['misc']) for c in start['deck']).items())
    return hashlib.sha256(json.dumps(cards, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def composition(start):
    return tuple(sorted(Counter(c['id'] for c in start['deck']).items()))


def distance(a, b):
    cards = [{c['id'] for c in s['deck']} for s in (a, b)]
    relics = [{r['id'] for r in s['relics']} for s in (a, b)]
    return sum(weight * (1 - len(x & y) / max(1, len(x | y)))
               for weight, (x, y) in zip((0.8, 0.2), (cards, relics)))


def main(source):
    old = Path('/home/sborowsk/project/sts/worktrees/nn-topologies/runs/schema=gumbel_nn_combat_v1/date=2026-10-30/id=gumbel-nn-champ-entity-ten')
    training = json.loads((old / 'out/inputs/starts.json').read_text())
    validation = read_jsonl(old / 'out/training/test/iter132/starts.jsonl')
    baselines = json.loads((old / 'out/inputs/baselines.json').read_text())
    cohorts = {f['fight_id']: 'original' for f in training}
    safe = {r['family'] for r in json.loads((source / 'scratch/champ-generalisation-audit/families.json').read_text()) if r['safe']}
    policies = json.loads((source / 'scratch/champ-generalisation-audit/policies.json').read_text())
    stores = ('champ-corpus-pilot-v1', 'human-deck-corpus', 'hdc-screen-w1', 'human-deck-library-v1')
    groups = defaultdict(dict)
    for policy in policies:
        if policy['kind'] != 'mcts20k' or not any(policy['run'].endswith('id=' + n) for n in stores):
            continue
        for filename in policy['files']:
            for row in pq.read_table(source / filename, columns=['fight_id', 'start', 'won', 'agent']).to_pylist():
                if row['agent'] != policy['agent']:
                    continue
                stable = {k: v for k, v in row['start'].items() if k not in ('seed', 'misc_rng', 'potion_rng')}
                key = (policy['run'], json.dumps(stable, sort_keys=True))
                groups[key][row['start']['seed']] = dict(row, source=str(source / filename))
    used = {composition(f['start']) for f in training}
    candidates = []
    for data in groups.values():
        rows = sorted(data.values(), key=lambda r: r['fight_id'])
        s, n, w = rows[0]['start'], len(rows), sum(r['won'] for r in rows)
        if (n >= 20 and 0.2 <= w / n <= 0.8 and family(s) in safe and composition(s) not in used
                and s['potions'] == [1, 1] and 8 <= len(s['deck']) <= 40
                and not {r['id'] for r in s['relics']} & {21, 57, 161}):  # Gambling Chip, Runic Dome, Prismatic Shard
            candidates.append(rows)
    selected = []
    for _ in range(15):
        available = [r for r in candidates if composition(r[0]['start']) not in used]
        chosen = max(available, key=lambda r: (min(distance(r[0]['start'], f['start']) for f in training),
                                               len(r), family(r[0]['start'])))
        s = chosen[0]['start']
        name = 'deck-' + family(s)[:8]
        base = dict(fight_id=name, start=s)
        training.append(base); used.add(composition(s)); cohorts[name] = 'added'
        historical = chosen[:50]
        for i, row in enumerate(historical):
            fid = f'champ25:{name}:validation:{i}'
            validation.append(dict(fight_id=fid, start=row['start']))
            baselines['mcts20k'][fid] = row['won']
        validation += fights(base, f'champ25:{name}:fresh', 50 - len(historical))
        w, n = sum(r['won'] for r in chosen), len(chosen)
        selected.append(dict(name=name, wins=w, n=n, wilson95=wilson(w, n), family=family(s),
                             source_files=sorted({r['source'] for r in chosen}),
                             historical_fights=[r['fight_id'] for r in historical]))
    assert len(training) == 25 and len(validation) == len({f['fight_id'] for f in validation}) == 1250
    assert set(Counter(f['fight_id'].split(':')[1] for f in validation).values()) == {50}
    out = Path('runs/benchmark/champ-25-inputs')
    out.mkdir(parents=True, exist_ok=False)
    for name, data in [('training.jsonl', training), ('validation.jsonl', validation)]:
        (out / name).write_text(''.join(json.dumps(r) + '\n' for r in data))
    write_json(out / 'baselines.json', baselines); write_json(out / 'cohorts.json', cohorts)
    write_json(out / 'selection.json', selected)
    print('Eligible scenarios:', len(candidates), '; selected:', [(d['name'], d['wins'], d['n']) for d in selected])


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
