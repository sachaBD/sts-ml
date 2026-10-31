"""Freeze a run's inputs, then launch expert_iteration.py on them. See README.md.

Everything the run reads is copied into OUT once: config.toml, inputs/{nn_play, starts.json, test.jsonl, ...}.
"""
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from apps.common.app import main, snapshot, write_json
from apps.gumbel_nn_combat_expert_iteration.config import load
from apps.gumbel_nn_combat_expert_iteration.reporting import deck_of
from environments.combat.starts import fights
from runs.run import run_dir


def inputs(raw):
    """Run ids this run reads (lineage for apps/common/launch.sh)."""
    evaluation = raw.get('evaluation', {})
    sources = {d['starts'] for d in raw.get('deck', [])}
    sources.update(evaluation[k] for k in ('source_run', 'starts', 'train_starts', 'cohorts', 'baselines', 'final_starts')
                   if evaluation.get(k))
    if raw.get('run', {}).get('warm_start_run'):
        sources.add(raw['run']['warm_start_run'])
    return sorted(sources)


def load_bases(decks):
    tables = {path: {r['deck_id']: r for r in pq.ParquetFile(path).read().to_pylist() if r['seed_kind'] == 'human'}
              for path in {d.starts for d in decks}}
    bases = {}
    for d in decks:
        start = tables[d.starts][d.deck_id]['start']
        if (start['hp'], start['max_hp']) != (d.hp, d.max_hp) or start['potions'] != [1, 1]:
            raise ValueError(f'{d.name}: configured HP/no-potion loadout mismatch')
        bases[d.name] = start
    return bases


def historical_tests(config, bases, frozen):
    """Evaluation on a historical teacher run's starts, plus generated starts for decks it lacks."""
    ev = config.evaluation
    source = run_dir(ev.source_run) / 'out' / ev.directory / 'results.jsonl'
    reference = [row for row in map(json.loads, source.read_text().splitlines()) if deck_of(row['fight_id']) in bases]
    stable = lambda start: {k: v for k, v in start.items() if k not in {'seed', 'misc_rng', 'potion_rng'}}
    for name, base in bases.items():
        rows = [row for row in reference if deck_of(row['fight_id']) == name]
        if not rows and ev.generated_per_deck:
            continue  # added decks have no historical teacher reference
        if len(rows) != ev.fights_per_deck or any(row['status'] != 'completed' for row in rows):
            raise ValueError(f'{name}: expected {ev.fights_per_deck} completed reference fights')
        if any(stable(row['fight']['start']) != stable(base) for row in rows):
            raise ValueError(f'{name}: reference loadout differs from training start')
    tests = [dict(fight_id=row['fight_id'], start=row['fight']['start']) for row in reference]
    originals = {deck_of(row['fight_id']) for row in reference}
    if ev.generated_per_deck:
        for name, base in bases.items():
            count = ev.generated_per_deck - sum(deck_of(row['fight_id']) == name for row in tests)
            if count < 0:
                raise ValueError('generated_per_deck must include the historical reference starts')
            tests += fights(dict(fight_id=name, start=base), f'{config.run.id}:{name}:validation', count)
        if len({row['start']['seed'] for row in tests}) != len(tests):
            raise ValueError('validation seed collision')
    write_json(frozen / 'reference.json', {row['fight_id']: row['fight']['won'] for row in reference})
    write_json(frozen / 'cohorts.json', {name: 'original' if name in originals else 'added' for name in bases})
    historical = dict(mcts20k={row['fight_id']: row['fight']['won'] for row in reference})
    puct = run_dir(ev.source_run) / 'out/update006/monitor/results.jsonl'
    if puct.exists():
        historical['puct'] = {row['fight_id']: row['fight']['won'] for row in map(json.loads, puct.read_text().splitlines())
                              if row['fight_id'] in historical['mcts20k'] and row['status'] == 'completed'}
    write_json(frozen / 'baselines.json', historical)
    return tests


def run(_raw, config_path, out):
    config = load(config_path.read_text())
    bases = load_bases(config.decks)
    frozen = out / 'inputs'
    frozen.mkdir(parents=True)  # refuse to overwrite an existing run's inputs
    snapshot(Path(config.run.worker), frozen, 'nn_play')
    (out / 'config.toml').write_bytes(config_path.read_bytes())
    write_json(frozen / 'starts.json', [dict(fight_id=name, start=start) for name, start in bases.items()])
    ev = config.evaluation
    if ev.starts:
        tests = [json.loads(line) for line in Path(ev.starts).read_text().splitlines()]
        for path, name in [(ev.cohorts, 'cohorts.json'), (ev.baselines, 'baselines.json'),
                           (ev.train_starts, 'test-in.jsonl'), (ev.final_starts, 'FINAL_TEST_DO_NOT_USE_FOR_SELECTION.jsonl')]:
            if path:
                snapshot(Path(path), frozen, name)
        write_json(frozen / 'reference.json', {})
    else:
        tests = historical_tests(config, bases, frozen)
    (frozen / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in tests))
    if config.run.warm_start_run:
        training = run_dir(config.run.warm_start_run) / 'out/training'
        rows = [json.loads(s) for s in (training / 'log.jsonl').read_text().splitlines()]
        latest = max(row['iteration'] for row in rows if 'iteration' in row)
        source = training / f'iter{latest:03d}/model.pt'
        _, digest = snapshot(source, frozen, 'initial-model.pt')
        write_json(frozen / 'warm-start.json', dict(source=str(source), sha256=digest, iteration=latest))

    command = [sys.executable, '-m', 'apps.gumbel_nn_combat_expert_iteration.expert_iteration', str(out.resolve())]
    write_json(out / 'command.json', command)
    if config.run.recovery_retries:
        from apps.gumbel_nn_combat_expert_iteration.supervisor import launch
        launch(command, out, retries=config.run.recovery_retries)
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main(run, inputs)
