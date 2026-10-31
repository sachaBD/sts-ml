"""Checkpoint-132 leaf-mix sweep. Default: read-only preflight; --run explicitly launches fights."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.common.app import read_jsonl, write_json as save
from apps.gumbel_nn_combat_expert_iteration.reporting import deck_of


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def outcomes(rows):
    result = {}
    for row in rows:
        key = row['fight_id']
        if key in result or type(row['won']) is not bool or row.get('capped', False):
            raise ValueError(f'duplicate, invalid or capped result: {key}')
        result[key] = row['won']
    return result


def verify(config):
    paths = {}
    for name, artifact in config['artifacts'].items():
        path = Path(artifact['path'])
        if digest(path) != artifact['sha256']:
            raise ValueError(f'artifact changed: {name}: {path}')
        paths[name] = path
    fights = read_jsonl(paths['fights'])
    keys = [r['fight_id'] for r in fights]
    decks = [deck_of(k) for k in keys]
    if len(set(keys)) != 500 or len(keys) != 500 or len(set(decks)) != 10 or any(decks.count(d) != 50 for d in decks):
        raise ValueError('expected exactly 500 unique fights, 50 per deck on ten decks')
    references = json.loads(paths['historical'].read_text())
    references['nn50'] = outcomes(read_jsonl(paths['nn50']))
    if any(set(v) != set(keys) or any(type(w) is not bool for w in v.values()) for v in references.values()):
        raise ValueError('reference outcomes do not exactly match the frozen fights')
    return paths, fights, references


def command(config, paths, mix, directory):
    w, s = config['workers'], config['search']
    args = [str(paths['worker']), 'play', str(paths['model.onnx']), str(directory / 'samples-0.bin')]
    flags = dict(threads=w['actors'], parallel=w['slots'], gpu_model=paths['model-gpu.onnx'],
                 gpu_batch=w['gpu_batch'], gpu_bank=','.join(str(paths[f'model-gpu-b{b}.onnx']) for b in w['gpu_bank']),
                 gpu_wait_us=w['gpu_wait_us'], timeout_ms=w['timeout_ms'], shutdown_timeout_ms=w['shutdown_timeout_ms'],
                 sims=s['sims'], particles=s['particles'], gumbel_scale=s['gumbel_scale'], rollout_mix=mix)
    for key, value in flags.items():
        args.extend(['--' + key.replace('_', '-'), str(value)])
    return args



def run(config, protocol, out):
    paths, fights, references = verify(config)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('commit changes before execution; benchmark source must be clean')
    out.mkdir(parents=True, exist_ok=False)  # never resume/overwrite a partial or completed attempt
    shutil.copy2(protocol, out / 'protocol.toml')
    shutil.copy2(paths['fights'], out / 'starts.jsonl')
    save(out / 'references.json', references)
    shutil.copy2(paths['baseline_log'], out / 'historical-nn50-worker.log')
    shutil.copy2(paths['training_args'], out / 'historical-training-args.json')
    shutil.copy2(paths['training_run'], out / 'historical-run.json')
    packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                         capture_output=True, text=True, check=True).stdout
    save(out / 'manifest.json', dict(revision=revision, protocol_sha256=digest(protocol), python=sys.version,
         executable=sys.executable, platform=platform.platform(), cpu=platform.processor(), gpu=gpu,
         runtime_environment={k:v for k,v in env.items() if k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                              'MKL_NUM_THREADS', 'LD_LIBRARY_PATH', 'ORT_TENSORRT_CACHE_PATH')},
         packages=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),
         historical_source=json.loads(paths['training_run'].read_text())['git'],
         provenance_note='Frozen historical binary; original source was dirty and its exact build is not reconstructible.'))
    for name, mix in config['variants'].items():
        verify(config)  # refuse changed artifacts, including between variants
        directory = out / name
        directory.mkdir()
        args = command(config, paths, mix, directory)
        save(directory / 'command.json', args)
        started = time.time()
        clock = time.monotonic()
        save(directory / 'status.json', dict(status='running', started_unix=started))
        try:
            with (out / 'starts.jsonl').open() as source, (directory / 'results.jsonl').open('w') as results, \
                    (directory / 'worker.log').open('w') as errors:
                subprocess.run(args, stdin=source, stdout=results, stderr=errors, env=env, check=True)
            measured = time.monotonic() - clock  # whole native process, including inference setup
            observed = outcomes(read_jsonl(directory / 'results.jsonl'))
            if set(observed) != set(references['nn50']):
                raise ValueError('incomplete or unexpected fight IDs')
            verify(config)
            save(directory / 'status.json', dict(status='complete', started_unix=started,
                 process_wall_seconds=measured, finished_unix=time.time()))
        except BaseException:
            save(directory / 'status.json', dict(status='failed', started_unix=started, finished_unix=time.time()))
            raise
    # Keep samples/cache/logs for inspection; no automatic deletion or hidden retry.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('protocol.toml'))
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.protocol.read_text())
    verify(config)
    if args.run:
        if args.out is None or not args.out.resolve().is_relative_to(ROOT / 'runs'):
            parser.error('--run requires a new --out directory under this worktree’s runs/')
        run(config, args.protocol, args.out.resolve())
    else:
        print('Verified frozen inputs: 500 fights, ten decks, checkpoint 132. No fights launched.')


if __name__ == '__main__':
    main()
