"""Builds the app's input files once, offline. A fights file is JSON lines, one playable fight per line:
{"fight_id": ..., "start": <combat_v4 start>}.

  python -m apps.gumbel_nn_combat_expert_iteration.make_fights decks DECKS.toml OUT.jsonl
      Training fights: one per [[deck]] (name, starts parquet, deck_id, hp, max_hp); fight_id is the deck name.

  python -m apps.gumbel_nn_combat_expert_iteration.make_fights teacher TRAINING.jsonl RUN:DIRECTORY OUT_DIR
          --name NAME --fights-per-deck N [--generated-per-deck M --namespace NS] [--baseline NAME=RUN:DIRECTORY ...]
      Evaluation on a teacher run's fights, for paired comparison: OUT_DIR/fights.jsonl, baselines.json (each
      player's result per fight it completed) and cohorts.json (decks with teacher fights are `original`,
      generated-only decks `added`). A player's results are RUN's out/DIRECTORY/results.jsonl.
"""
import argparse
import json
from pathlib import Path
import tomllib

import pyarrow.parquet as pq

from apps.common.app import read_jsonl
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
    return [dict(fight_id=name, start=row['start']) for name, row in load_bases(decks).items()]


def results(player):
    """RUN:DIRECTORY -> that run's results rows."""
    run, directory = player.split(':', 1)
    return read_jsonl(run_dir(run) / 'out' / directory / 'results.jsonl')


def teacher_fights(training, teacher, name, fights_per_deck, generated_per_deck=0, namespace=None, baselines=()):
    """(fights, baselines, cohorts): the teacher's fights for the training decks, topped up to
    generated_per_deck per deck with fresh seeds; baselines {name: {fight_id: won}} over those fights."""
    starts = {f['fight_id']: f['start'] for f in training}
    reference = [row for row in results(teacher) if deck_of(row['fight_id']) in starts]
    stable = lambda start: {k: v for k, v in start.items() if k not in {'seed', 'misc_rng', 'potion_rng'}}
    for deck, start in starts.items():
        rows = [row for row in reference if deck_of(row['fight_id']) == deck]
        if not rows and generated_per_deck:
            continue  # added decks have no teacher fights
        if len(rows) != fights_per_deck or any(row['status'] != 'completed' for row in rows):
            raise ValueError(f'{deck}: expected {fights_per_deck} completed teacher fights')
        if any(stable(row['fight']['start']) != stable(start) for row in rows):
            raise ValueError(f'{deck}: teacher loadout differs from training start')
    out = [dict(fight_id=row['fight_id'], start=row['fight']['start']) for row in reference]
    originals = {deck_of(row['fight_id']) for row in reference}
    if generated_per_deck:
        for deck, start in starts.items():
            count = generated_per_deck - sum(deck_of(f['fight_id']) == deck for f in out)
            if count < 0:
                raise ValueError('generated_per_deck must include the teacher fights')
            out += fights(dict(fight_id=deck, start=start), f'{namespace}:{deck}:validation', count)
        if len({f['start']['seed'] for f in out}) != len(out):
            raise ValueError('validation seed collision')
    won = {name: {row['fight_id']: row['fight']['won'] for row in reference}}
    for other, player in baselines:
        won[other] = {row['fight_id']: row['fight']['won'] for row in results(player)
                      if row['fight_id'] in won[name] and row['status'] == 'completed'}
    cohorts = {deck: 'original' if deck in originals else 'added' for deck in starts}
    return out, won, cohorts


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
    t.add_argument('teacher', help='RUN:DIRECTORY of the teacher results')
    t.add_argument('out', type=Path)
    t.add_argument('--name', required=True, help="the teacher's baseline name")
    t.add_argument('--fights-per-deck', type=int, required=True)
    t.add_argument('--generated-per-deck', type=int, default=0)
    t.add_argument('--namespace', help='seed namespace of generated fights (required with --generated-per-deck)')
    t.add_argument('--baseline', action='append', default=[], metavar='NAME=RUN:DIRECTORY',
                   help='another player on the same fights (repeatable)')
    a = p.parse_args()
    if a.command == 'decks':
        write_fights(a.out, deck_fights(tomllib.loads(a.decks.read_text())['deck']))
        return
    if a.generated_per_deck and not a.namespace:
        p.error('--generated-per-deck needs --namespace')
    rows, baselines, cohorts = teacher_fights(read_jsonl(a.training), a.teacher, a.name, a.fights_per_deck,
                                              a.generated_per_deck, a.namespace,
                                              [b.split('=', 1) for b in a.baseline])
    a.out.mkdir(parents=True, exist_ok=True)
    write_fights(a.out / 'fights.jsonl', rows)
    (a.out / 'baselines.json').write_text(json.dumps(baselines, indent=2) + '\n')
    (a.out / 'cohorts.json').write_text(json.dumps(cohorts, indent=2) + '\n')


if __name__ == '__main__':
    main()
