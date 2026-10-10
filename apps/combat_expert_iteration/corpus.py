"""Resumable four-round corpus pilot. Run --preflight-only before the full controller.

All protocol choices live in frozen manifests. Existing fixed-deck entry points are unchanged.
One exclusive run lock; full-start cache identity; completed games survive interruptions.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq

from apps.common.app import sha256
from .corpus_io import atomic_json, freeze_json, identity, journal_rows, play_stage
from .corpus_manifest import load_manifest, collection_plan, validate_production_freeze
from .expert_iteration import make_starts


class Corpus:
    def __init__(self, run, workers=10):
        self.run = Path(run).resolve()
        self.out = self.run / 'out'
        self.workers = workers
        self.prepared = json.loads((self.out / 'prepared.json').read_text())
        self.manifest = load_manifest(self.out / 'families.json')
        validate_production_freeze(self.manifest)
        self.families = json.loads((self.out / 'families.json').read_text())['families']
        self.bases = json.loads((self.out / 'bases.json').read_text())
        self.worker = self.prepared['worker']
        self.initial = Path(self.prepared['model'])
        self.legacy = json.loads((self.out / 'legacy-shards.json').read_text())
        self.used_seeds = {v['row']['start']['seed'] for v in self.bases.values()}
        # Guard the exact old binary, model and shards; no rebuilt worker is substituted.
        for path, digest in self.manifest.legacy_run['artifact_sha256'].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f'legacy artifact changed: {path}')
        sources = [Path(__file__), *Path('apps/combat_expert_iteration').glob('corpus_*.py'),
                   Path('agents/combat/pv/corpus_replay.py'), Path('agents/combat/pv/train_corpus.py'),
                   Path('agents/combat/pv/data.py'), Path('agents/combat/pv/model.py'), Path('agents/combat/pv/train.py')]
        freeze_json(self.out / 'code-hashes.json', {str(p.resolve()): sha256(p) for p in sorted(set(sources))})
        freeze_json(self.out / 'execution.json', dict(workers=workers, sims=2000, rollout_mix=.5,
                                                     teacher_sims=20000, rounds=4, protocol_sha256=sha256(Path('experiments/multi-fight-champ-expert/CORPUS_PILOT.md'))))
        self.state = dict(status='running', pid=os.getpid(), stage='initializing', updated=time.time())

    def status(self, stage, progress=None, **extra):
        self.state.update(stage=stage, progress=progress, updated=time.time(), **extra)
        atomic_json(self.out / 'status.json', self.state)

    def starts(self, label, allocations):
        path = self.out / 'starts' / f'{label}.json'
        if path.exists():
            rows = json.loads(path.read_text())
            if Counter(r['fight_id'].split(':')[1] for r in rows) != Counter(allocations):
                raise ValueError(f'cached starts allocation changed: {label}')
            self.used_seeds.update(r['start']['seed'] for r in rows)
            return rows
        rows = []
        for fid, n in sorted(allocations.items()):
            role = self.manifest.by_id[fid].role
            if role == 'final':
                raise ValueError('final families cannot be played by pilot controller')
            rows.extend(make_starts(self.bases[fid]['row'], f'{self.run.name.removeprefix("id=")}:{fid}',
                                    label, n, self.used_seeds))
        freeze_json(path, rows)
        return rows

    def command(self, model=None, collect=False, sims=None):
        if model is None:
            return [self.worker, 'teacher', str(sims or 20000)]
        command = [self.worker, 'play', str(Path(model) / 'model.onnx'), str(sims or 2000), '--rollout-mix', '0.5']
        if collect:
            command += ['--explore', '--sample-turns']
        return command

    def play(self, name, starts, model=None, collect=False, sims=None):
        self.status(name)
        print(f'{time.strftime("%FT%T")} {name}: {len(starts)} games', flush=True)
        result = play_stage(self.out / name, starts, self.command(model, collect, sims), self.workers,
                            lambda progress: self.status(name, progress))
        if collect and any(r['status'] != 'completed' for r in result.values()):
            raise RuntimeError(f'{name}: capped collection lacks actual terminal labels; stop for review, do not silently omit')
        return result

    def call(self, name, command, log):
        self.status(name)
        print(f'{time.strftime("%FT%T")} {name}', flush=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('w') as stream:
            subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)

    def encode(self, name, source, round_number):
        directory = self.out / name
        rows = directory / 'rows' / 'rows.parquet'
        signature = dict(fights=sha256(directory / 'fights.parquet'), search=sha256(directory / 'search.parquet'),
                         worker=sha256(Path(self.worker)))
        freeze_json(directory / 'encode-inputs.json', signature)
        if not rows.exists():
            self.call(f'encode {name}', [sys.executable, '-m', 'agents.combat.pv.data', '--fights', directory / 'fights.parquet',
                       '--search', directory / 'search.parquet', '--encounters', '39', '--worker', self.worker,
                       '--out', rows.parent], directory / 'encode.log')
        ids = set(pq.ParquetFile(rows).read(columns=['fight_id'])['fight_id'].to_pylist())
        completed = {fid for fid, r in journal_rows(directory / 'results.jsonl').items() if r['status'] == 'completed'}
        if ids != completed:
            raise ValueError(f'encoded fight coverage mismatch: {name}')
        spec = dict(path=str(rows), sha256=sha256(rows), fights={fid: dict(family_id=fid.split(':')[1], source=source, round=round_number) for fid in sorted(ids)})
        freeze_json(directory / 'shard.json', spec)
        return spec

    def training(self, round_number, model, shards, smoke=False):
        directory = self.out / ('smoke-training' if smoke else f'round{round_number}')
        directory.mkdir(parents=True, exist_ok=True)
        config = dict(init=str(model / 'model.pt'), init_sha256=sha256(model / 'model.pt'), families=self.families,
                      round=round_number, shards=shards, seed=20261009 + round_number * 100)
        freeze_json(directory / 'train.json', config)
        final = directory / 'model'
        if (final / 'complete.json').exists():
            complete = json.loads((final / 'complete.json').read_text())
            if complete['checkpoint_sha256'] != sha256(final / 'model.pt') or complete['onnx_sha256'] != sha256(final / 'model.onnx'):
                raise ValueError('published model hash mismatch')
            return final
        partial = directory / 'model.partial'
        if partial.exists():
            raise ValueError(f'partial training requires review before retry: {partial}')
        command = [sys.executable, '-m', 'agents.combat.pv.train_corpus', directory / 'train.json', '--out', partial]
        if smoke:
            command += ['--smoke-steps', '2']
        self.call(f'train {directory.name}', command, directory / 'train.log')
        if not (partial / 'complete.json').exists():
            raise ValueError('trainer did not publish completion marker')
        os.replace(partial, final)
        return final

    def preflight(self):
        if (self.out / 'preflight-complete.json').exists():
            return
        # Support-only games: no outcome-based selection/replacement and never training/evaluation evidence.
        supported = {f.family_id: 1 for f in self.manifest.families if f.role in ('pilot', 'development')}
        starts = self.starts('smoke-support', supported)
        results = self.play('smoke-support', starts, self.initial, sims=1)
        if any(r['status'] != 'completed' for r in results.values()):
            raise RuntimeError('support smoke did not complete; review before launching')
        first = {f.family_id: 1 for f in self.manifest.families if f.role == 'pilot' and f.wave == 1}
        self.play('smoke-teacher', self.starts('smoke-teacher', first), sims=1)
        # Separate learner smoke shard limited to wave one, so later families never enter round-one replay.
        self.play('smoke-learner', self.starts('smoke-learner', first), self.initial, sims=1)
        t = self.encode('smoke-teacher', 'teacher', 1)
        l = self.encode('smoke-learner', 'learner', 1)
        # Canonical encoder also checks support for every selected train/dev deck.
        self.encode('smoke-support', 'learner', 1)
        candidate = self.training(1, self.initial, self.legacy + [t, l], smoke=True)
        original = next(f.family_id for f in self.manifest.families if f.role == 'original')
        self.play('smoke-export', self.starts('smoke-export', {original: 1}), candidate, sims=1)
        atomic_json(self.out / 'preflight-complete.json', dict(time=time.time(), note='81 tiny smoke games, 2 optimizer steps; smoke data excluded from production'))
        self.status('preflight passed', status='ready')

    def evaluate(self, round_number, model, starts, teacher):
        result = self.play(f'dev-round{round_number}', starts, model)
        baseline = journal_rows(self.out / 'dev-round0' / 'results.jsonl') if round_number else result
        def won(r):
            if r['status'] not in ('completed', 'capped'):
                raise ValueError('unresolved evaluation infrastructure error')
            return int(r['status'] == 'completed' and r['fight']['won'])
        families = sorted({r['fight_id'].split(':')[1] for r in starts})
        diffs, incumbent = [], []
        for family in families:
            ids = [r['fight_id'] for r in starts if r['fight_id'].split(':')[1] == family]
            diffs.append(np.mean([won(result[i]) - won(teacher[i]) for i in ids]))
            incumbent.append(np.mean([won(result[i]) - won(baseline[i]) for i in ids]))
        rng = np.random.default_rng(20261009)
        idx = rng.integers(len(families), size=(10000, len(families)))
        ci = np.quantile(np.asarray(diffs)[idx].mean(1), [.025, .975]).tolist()
        row = dict(round=round_number, n=len(result), wins=sum(won(r) for r in result.values()),
                   teacher_wins=sum(won(r) for r in teacher.values()), gap=float(np.mean(diffs)), family_ci95=ci,
                   gap_vs_initial=float(np.mean(incumbent)), initial_ci95=np.quantile(np.asarray(incumbent)[idx].mean(1), [.025, .975]).tolist(),
                   per_family=dict(zip(families, map(float, diffs))), statuses=dict(Counter(r['status'] for r in result.values())))
        path = self.out / 'curve.json'
        checks = json.loads(path.read_text())['checks'] if path.exists() else []
        checks = [r for r in checks if r['round'] != round_number] + [row]
        atomic_json(path, dict(checks=sorted(checks, key=lambda r: r['round'])))
        print(json.dumps(dict(event='development', **row)), flush=True)

    def run_rounds(self):
        self.preflight()
        self.state['status'] = 'running'
        dev = self.starts('development', {f.family_id: 20 for f in self.manifest.families if f.role == 'development'})
        teacher = self.play('dev-teacher', dev)
        self.evaluate(0, self.initial, dev, teacher)
        model, shards = self.initial, list(self.legacy)
        for k in range(1, 5):
            plan = collection_plan(self.manifest, k, seed=20261009)
            for source in ('teacher', 'learner', 'revisit_learner'):
                allocations = {j['family_id']: j['games'] for j in plan if j['source'] == source}
                if not allocations:
                    continue
                name = f'round{k}-{source}'
                starts = self.starts(name, allocations)
                self.play(name, starts, None if source == 'teacher' else model, collect=source != 'teacher')
                if any(r['status'] != 'completed' for r in journal_rows(self.out / name / 'results.jsonl').values()):
                    raise RuntimeError('nonterminal teacher collection requires review')
                shards.append(self.encode(name, 'teacher' if source == 'teacher' else 'learner', k))
            model = self.training(k, model, shards)
            if k in (1, 4):
                self.evaluate(k, model, dev, teacher)
        self.status('four rounds complete', status='completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error('workers must be 1..12')
    args.run.joinpath('out').mkdir(parents=True, exist_ok=True)
    with (args.run / 'out/controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = None
        try:
            controller = Corpus(args.run, args.workers)
            if args.preflight_only:
                controller.preflight()
            else:
                controller.run_rounds()
        except BaseException as error:
            if controller:
                controller.status(controller.state['stage'], status='failed', error=str(error))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
