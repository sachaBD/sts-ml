"""Builds the app's input files once, offline. A fights file is JSON lines, one playable fight per line:
{"fight_id": ..., "start": <combat_v4 start>}.

  python -m apps.gumbel_nn_combat_expert_iteration.make_fights decks DECKS.toml OUT.jsonl
      Training fights: one per [[deck]] (name, starts parquet, deck_id, hp, max_hp); fight_id is the deck name.

  python -m apps.gumbel_nn_combat_expert_iteration.make_fights teacher TRAINING.jsonl SOURCE_RUN DIRECTORY OUT_DIR
          --fights-per-deck N [--generated-per-deck M --namespace NS]
      Evaluation on an old teacher run's starts, for paired comparison: OUT_DIR/fights.jsonl, baselines.json
      (the teacher's result per fight, plus PUCT where recorded) and cohorts.json (decks with teacher fights are
      `original`, generated-only decks `added`).
"""
import argparse
import json
from pathlib import Path
import re
import tomllib

import pyarrow.parquet as pq

from environments.combat.starts import fights
from runs.run import run_dir

from .reporting import deck_of


def load_bases(decks):
    """{deck name: corpus row} for [[deck]] tables, checking each deck's HP and empty potion slots."""
    tables = {path: {r['deck_id']: r for r in pq.ParquetFile(path).read().to_pylist() if r['seed_kind'] == 'human'}
              for path in {d['starts'] for d in decks}}
    bases = {}
    for d in decks:
        row = tables[d['starts']][d['deck_id']]
        start = row['start']
        if (start['hp'], start['max_hp']) != (d['hp'], d['max_hp']) or start['potions'] != [1, 1]:
            raise ValueError(f'{d["name"]}: configured HP/no-potion loadout mismatch')
        bases[d['name']] = row
    return bases


def deck_fights(decks):
    names = [d['name'] for d in decks]
    if len(set(names)) != len(names) or not all(re.fullmatch(r'[a-z0-9_-]+', n) for n in names):
        raise ValueError('deck names must be unique and use [a-z0-9_-]')
    return [dict(fight_id=name, start=row['start']) for name, row in load_bases(decks).items()]


def teacher_fights(training, source_run, directory, fights_per_deck, generated_per_deck=0, namespace=None):
    """(fights, baselines, cohorts) on the teacher's starts for the training decks."""
    starts = {f['fight_id']: f['start'] for f in training}
    source = run_dir(source_run) / 'out' / directory / 'results.jsonl'
    reference = [row for row in map(json.loads, source.read_text().splitlines()) if deck_of(row['fight_id']) in starts]
    stable = lambda start: {k: v for k, v in start.items() if k not in {'seed', 'misc_rng', 'potion_rng'}}
    for name, start in starts.items():
        rows = [row for row in reference if deck_of(row['fight_id']) == name]
        if not rows and generated_per_deck:
            continue  # added decks have no teacher reference
        if len(rows) != fights_per_deck or any(row['status'] != 'completed' for row in rows):
            raise ValueError(f'{name}: expected {fights_per_deck} completed reference fights')
        if any(stable(row['fight']['start']) != stable(start) for row in rows):
            raise ValueError(f'{name}: reference loadout differs from training start')
    out = [dict(fight_id=row['fight_id'], start=row['fight']['start']) for row in reference]
    originals = {deck_of(row['fight_id']) for row in reference}
    if generated_per_deck:
        for name, start in starts.items():
            count = generated_per_deck - sum(deck_of(f['fight_id']) == name for f in out)
            if count < 0:
                raise ValueError('generated_per_deck must include the teacher reference starts')
            out += fights(dict(fight_id=name, start=start), f'{namespace}:{name}:validation', count)
        if len({f['start']['seed'] for f in out}) != len(out):
            raise ValueError('validation seed collision')
    baselines = dict(mcts20k={row['fight_id']: row['fight']['won'] for row in reference})
    puct = run_dir(source_run) / 'out/update006/monitor/results.jsonl'
    if puct.exists():
        baselines['puct'] = {row['fight_id']: row['fight']['won'] for row in map(json.loads, puct.read_text().splitlines())
                             if row['fight_id'] in baselines['mcts20k'] and row['status'] == 'completed'}
    cohorts = {name: 'original' if name in originals else 'added' for name in starts}
    return out, baselines, cohorts


def write_fights(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('decks')
    d.add_argument('decks', type=Path)
    d.add_argument('out', type=Path)
    t = sub.add_parser('teacher')
    t.add_argument('training', type=Path)
    t.add_argument('source_run')
    t.add_argument('directory')
    t.add_argument('out', type=Path)
    t.add_argument('--fights-per-deck', type=int, required=True)
    t.add_argument('--generated-per-deck', type=int, default=0)
    t.add_argument('--namespace', help='seed namespace of generated fights (required with --generated-per-deck)')
    a = p.parse_args()
    if a.command == 'decks':
        write_fights(a.out, deck_fights(tomllib.loads(a.decks.read_text())['deck']))
        return
    if a.generated_per_deck and not a.namespace:
        p.error('--generated-per-deck needs --namespace')
    training = [json.loads(line) for line in a.training.read_text().splitlines() if line.strip()]
    rows, baselines, cohorts = teacher_fights(training, a.source_run, a.directory, a.fights_per_deck,
                                              a.generated_per_deck, a.namespace)
    a.out.mkdir(parents=True, exist_ok=True)
    write_fights(a.out / 'fights.jsonl', rows)
    (a.out / 'baselines.json').write_text(json.dumps(baselines, indent=2) + '\n')
    (a.out / 'cohorts.json').write_text(json.dumps(cohorts, indent=2) + '\n')


if __name__ == '__main__':
    main()
