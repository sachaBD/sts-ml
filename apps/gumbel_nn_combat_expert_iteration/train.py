"""Load fixed combat starts and launch the infra-backed Gumbel NN loop. See README.md."""
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from apps.common.app import main, snapshot, write_json
from apps.combat_expert_iteration.expert_iteration import check_decks, deck_of
from runs.run import run_dir


def inputs(config):
    sources = {d['starts'] for d in config['deck']}
    sources.update(v for k, v in config['evaluation'].items() if k in {'source_run', 'starts', 'train_starts', 'cohorts', 'baselines', 'final_starts'} and v)
    if config['run'].get('warm_start_run'):
        sources.add(config['run']['warm_start_run'])
    return sorted(sources)


def load_bases(decks):
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


def run(config, config_path, out):
    r, ev, decks = config['run'], config['evaluation'], config['deck']
    check_decks(decks)
    bases = load_bases(decks)
    reference = []
    if not ev.get('starts'):
        source = run_dir(ev['source_run']) / 'out' / ev['directory'] / 'results.jsonl'
        reference = [json.loads(line) for line in source.read_text().splitlines()]
        reference = [row for row in reference if deck_of(row['fight_id']) in bases]
        for name, base in bases.items():
            rows = [row for row in reference if deck_of(row['fight_id']) == name]
            if not rows and ev.get('generated_per_deck'):
                continue  # added decks have no historical teacher reference
            if len(rows) != ev['fights_per_deck'] or any(row['status'] != 'completed' for row in rows):
                raise ValueError(f'{name}: expected {ev["fights_per_deck"]} completed reference fights')
            stable = lambda start: {k: v for k, v in start.items() if k not in {'seed', 'misc_rng', 'potion_rng'}}
            if any(stable(row['fight']['start']) != stable(base['start']) for row in rows):
                raise ValueError(f'{name}: reference loadout differs from training start')

    frozen = out / 'inputs'
    frozen.mkdir(parents=True)  # refuse to overwrite an existing run's inputs
    worker, _ = snapshot(Path(r['worker']), frozen, 'nn_play')
    (out / 'config.toml').write_bytes(config_path.read_bytes())
    write_json(frozen / 'starts.json', [dict(fight_id=name, start=row['start']) for name, row in bases.items()])
    if ev.get('starts'):
        tests = [json.loads(line) for line in Path(ev['starts']).read_text().splitlines()]
        (frozen / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in tests))
        for key, name in [('cohorts', 'cohorts.json'), ('baselines', 'baselines.json'),
                          ('train_starts', 'test-in.jsonl'), ('final_starts', 'FINAL_TEST_DO_NOT_USE_FOR_SELECTION.jsonl')]:
            if ev.get(key):
                snapshot(Path(ev[key]), frozen, name)
        write_json(frozen / 'reference.json', {})
    else:
        tests = [dict(fight_id=row['fight_id'], start=row['fight']['start']) for row in reference]
        originals = {deck_of(row['fight_id']) for row in reference}
        if ev.get('generated_per_deck'):
            from agents.combat.nn.loop import fights
            for name, base in bases.items():
                existing = sum(deck_of(row['fight_id']) == name for row in tests)
                count = ev['generated_per_deck'] - existing
                if count < 0:
                    raise ValueError('generated_per_deck must include the historical reference starts')
                tests += fights(base, f'{r["id"]}:{name}:validation', count)
            if len({row['start']['seed'] for row in tests}) != len(tests):
                raise ValueError('validation seed collision')
        (frozen / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in tests))
        write_json(frozen / 'reference.json', {row['fight_id']: row['fight']['won'] for row in reference})
        write_json(frozen / 'cohorts.json', {name: 'original' if name in originals else 'added' for name in bases})
        historical = dict(mcts20k={row['fight_id']: row['fight']['won'] for row in reference})
        puct = run_dir(ev['source_run']) / 'out/update006/monitor/results.jsonl'
        if puct.exists():
            historical['puct'] = {row['fight_id']: row['fight']['won'] for row in map(json.loads, puct.read_text().splitlines())
                                  if row['fight_id'] in historical['mcts20k'] and row['status'] == 'completed'}
        write_json(frozen / 'baselines.json', historical)
    loop = dict(config['loop'])
    if r.get('warm_start_run'):
        training = run_dir(r['warm_start_run']) / 'out/training'
        rows = [json.loads(s) for s in (training / 'log.jsonl').read_text().splitlines()]
        latest = max(row['iteration'] for row in rows if 'iteration' in row)
        source = training / f'iter{latest:03d}/model.pt'
        checkpoint, digest = snapshot(source, frozen, 'initial-model.pt')
        loop['initial_checkpoint'] = str(checkpoint)
        write_json(frozen / 'warm-start.json', dict(source=str(source), sha256=digest, iteration=latest))

    command = [sys.executable, '-m', 'agents.combat.nn.loop',
               '--start', str((frozen / 'starts.json').resolve()),
               '--test', str((frozen / 'test.jsonl').resolve()),
               '--test-at', ','.join(map(str, ev['at'])), '--play', str(worker),
               '--out', str((out / 'training').resolve()),
               '--fight-namespace', r['seed_namespace'], '--games', str(r['games_per_deck'] * len(decks))]
    if ev.get('train_starts'):
        command += ['--test-in', str((frozen / 'test-in.jsonl').resolve())]
    for key, value in loop.items():
        flag = '--' + key.replace('_', '-')
        command += ([flag] if value else []) if isinstance(value, bool) else [flag, str(value)]
    write_json(out / 'command.json', command)
    if r.get('recovery_retries', 0):
        from apps.gumbel_nn_combat_expert_iteration.supervisor import launch
        launch(command, out, retries=r['recovery_retries'])
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main(run, inputs)
