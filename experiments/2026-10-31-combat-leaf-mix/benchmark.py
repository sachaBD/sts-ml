"""Check frozen inputs; --run OUT runs leaf mixes, or --sim-sweep [N ...] runs NN100 budgets."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

PROTOCOL = Path(__file__).with_name('protocol.toml')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outcomes(path):
    data = rows(path)
    result = {r['fight_id']: r['won'] for r in data}
    if len(result) != len(data) or any(type(r['won']) is not bool or r.get('capped') for r in data):
        raise ValueError(f'duplicate, invalid or capped results: {path}')
    return result


def preflight(config):
    root = Path(config['run'])
    for name, expected in config['sha256'].items():
        with (root / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'changed artifact: {name}')
    test = root / 'out/training/test/iter132'
    fights = rows(test / 'starts.jsonl')
    ids = {r['fight_id'] for r in fights}
    decks = Counter(r['fight_id'].split(':')[1] for r in fights)
    if len(ids) != 500 or len(fights) != 500 or len(decks) != 10 or set(decks.values()) != {50}:
        raise ValueError('expected 500 unique starts, 50 per deck')
    references = json.loads((root / 'out/inputs/baselines.json').read_text())
    references['nn50'] = outcomes(test / 'results.jsonl')
    if any(set(v) != ids or any(type(w) is not bool for w in v.values()) for v in references.values()):
        raise ValueError('baseline IDs/outcomes do not match starts')
    print('Preflight passed: frozen checkpoint 132, 500 matching fights, ten decks.')
    return root, test / 'starts.jsonl', ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, metavar='OUT', help='launch fights into a new directory under runs/')
    parser.add_argument('--sim-sweep', nargs='*', type=int, metavar='N', help='NN100 budgets (default: 16 32 64 128)')
    args = parser.parse_args()
    root, starts, ids = preflight(tomllib.loads(PROTOCOL.read_text()))
    if args.run is None:
        return
    repo = Path(__file__).resolve().parents[2]
    out = args.run.resolve()
    if not out.is_relative_to(repo / 'runs'):
        parser.error('output must be under this worktree’s runs/')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        parser.error('commit changes before execution')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(PROTOCOL, out / 'protocol.toml')
    (out / 'revision.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True))
    packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    model = root / 'out/training/iter132'
    variants = ([(f'sims{s}', str(s), '0') for s in (args.sim_sweep or (16, 32, 64, 128))] if args.sim_sweep is not None
                else [('nn75', '256', '0.25'), ('nn100', '256', '0')])
    for name, sims, mix in variants:
        stage = out / name
        stage.mkdir()
        command = [str(root / 'out/inputs/nn_play'), 'play', str(model / 'model.onnx'), str(stage / 'samples.bin'),
                   '--gpu-model', str(model / 'model-gpu.onnx'), '--gpu-batch', '512',
                   '--gpu-bank', ','.join(str(model / f'model-gpu-b{b}.onnx') for b in (256, 32)),
                   '--threads', '9', '--parallel', '224', '--gpu-wait-us', '100',
                   '--timeout-ms', '60000', '--shutdown-timeout-ms', '5000',
                   '--sims', sims, '--particles', '8', '--gumbel-scale', '0', '--rollout-mix', mix]
        (stage / 'command.txt').write_text(shlex.join(command) + '\n')
        with starts.open() as source, (stage / 'results.jsonl').open('w') as results, (stage / 'worker.log').open('w') as log:
            before = time.monotonic()
            subprocess.run(command, stdin=source, stdout=results, stderr=log, env=env, check=True)
            elapsed = time.monotonic() - before
        if set(outcomes(stage / 'results.jsonl')) != ids:
            raise ValueError(f'incomplete results: {stage}')
        (stage / 'wall_seconds.txt').write_text(f'{elapsed:.3f}\n')


if __name__ == '__main__':
    main()
