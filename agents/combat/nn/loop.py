"""Gumbel self-play loop: (infra actors -> infra readers + learner -> export) x N, with tests.

  python -m agents.combat.nn.loop --start START.json --out DIR [--test FIGHTS.jsonl --test-at 0,3,10]

START.json: one fight {"fight_id", "start"}, or a list of named fights (combat_v4 starts).
Training reuses the starts with new seeds; --games is the total, balanced across a list.
Writes DIR/iterNNN/{samples-*.bin, results.jsonl, model.pt, model.onnx}, DIR/test/iterNNN/results.jsonl,
DIR/log.jsonl (one line per iteration or test) and DIR/status.json (live stage and progress).
"""
import argparse
import copy
import fcntl
import itertools
import shutil
import hashlib
import json
import subprocess
import os
import sys
import time
from pathlib import Path

import torch

from .graphs import Exporter
from .replay import Replay
from .model import Net, export, export_static
from .metrics import DeviceSampler
from .train import train

PLAY = Path(__file__).resolve().parents[3] / 'build/nn/agents/combat/nn/nn_play'
MASK = (1 << 64) - 1


def rng_state(seed):
    """sts::Random(seed) before any draw (libgdx RandomXS128 seeding)."""
    def murmur(x):
        x ^= x >> 33; x = x * (-49064778989728563 & MASK) & MASK
        x ^= x >> 33; x = x * (-4265267296055464877 & MASK) & MASK
        return (x ^ (x >> 33)) & MASK
    first = murmur(seed if seed else 1 << 63)
    return {'counter': 0, 'seed0': first, 'seed1': murmur(first)}


def fights(base, name, n):
    """n copies of base with fresh seeds derived from name."""
    if isinstance(base, list):
        if not base or n % len(base):
            raise ValueError('games must be a multiple of the nonempty start list length')
        names = [f['fight_id'] for f in base]
        if len(set(names)) != len(names):
            raise ValueError('start list fight_ids must be unique')
        # Interleave decks so collection/draining does not run one deck at a time.
        decks = [fights(f, f'{name}:{f["fight_id"]}', n // len(base)) for f in base]
        return [decks[d][i] for i in range(n // len(base)) for d in range(len(base))]
    out = []
    for i in range(n):
        seed = int.from_bytes(hashlib.sha256(f'{name}:{i}'.encode()).digest()[:8], 'little')
        f = copy.deepcopy(base)
        f['fight_id'] = f'{name}:{i}'
        f['start'].update(seed=seed, misc_rng=rng_state(seed), potion_rng=rng_state(seed))
        out.append(f)
    return out


class Run:
    def __init__(self, out, gpu_batch=512, gpu_bank=(), gpu_wait_us=100,
                 parallel=224, timeout_ms=60000, shutdown_timeout_ms=5000,
                 sample_devices=True, play_path=PLAY):
        self.out = out
        self.play_path = play_path
        self.gpu_batch = gpu_batch
        self.gpu_bank, self.gpu_wait_us = gpu_bank, gpu_wait_us
        self.parallel = parallel
        self.timeout_ms, self.shutdown_timeout_ms = timeout_ms, shutdown_timeout_ms
        self.sample_devices = sample_devices

    def status(self, **fields):
        tmp = self.out / '.status.json'
        tmp.write_text(json.dumps(dict(time=time.time(), pid=os.getpid(), **fields)))
        tmp.replace(self.out / 'status.json')

    def log(self, **row):
        with (self.out / 'log.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def play(self, model, games, directory, sims, gumbel_scale, rollout_mix, workers, stage):
        """One process: CPU worker threads feed the shared GPU. Results are journaled as games finish."""
        directory.mkdir(parents=True, exist_ok=True)
        sample = directory / 'samples-0.bin'
        if sample.exists():
            raise FileExistsError(f'{sample}: refusing to append duplicate gameplay')
        starts = directory / 'starts.jsonl'
        starts.write_text(''.join(json.dumps(g) + '\n' for g in games))
        command = [str(self.play_path), 'play', str(model), str(sample), '--threads', str(workers), '--parallel', str(self.parallel),
                   '--gpu-model', str(model.with_name('model-gpu.onnx')), '--gpu-batch', str(self.gpu_batch),
                   '--sims', str(sims), '--gumbel-scale', str(gumbel_scale),
                   '--rollout-mix', str(rollout_mix), '--gpu-wait-us', str(self.gpu_wait_us),
                   '--timeout-ms', str(self.timeout_ms), '--shutdown-timeout-ms', str(self.shutdown_timeout_ms)]
        if self.gpu_bank:
            command += ['--gpu-bank', ','.join(str(model.with_name(f'model-gpu-b{b}.onnx')) for b in self.gpu_bank)]
        # TensorRT and CUDA libraries come from the dependency-managed environment, not system-wide installs.
        packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        results = []
        self.status(stage=stage, phase='play', done=0, total=len(games), directory=str(directory))
        with DeviceSampler(directory / 'gpu.csv', enabled=self.sample_devices), \
                starts.open() as source, (directory / 'results.jsonl').open('w') as journal, (directory / 'worker.log').open('w') as errors:
            with subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=errors, text=True, env=env) as process:
                for line in process.stdout:
                    results.append(json.loads(line)); journal.write(line); journal.flush()
                    self.status(stage=stage, phase='play', done=len(results), total=len(games),
                                child_pid=process.pid, directory=str(directory))
            if process.returncode:
                raise RuntimeError(f'nn_play exited {process.returncode}; see {directory / "worker.log"}')
        if len(results) != len(games):
            raise RuntimeError(f'nn_play returned {len(results)} of {len(games)} games')
        return results


def collection_stats(results):
    times = sorted(r['finished_unix'] for r in results if 'finished_unix' in r)
    return dict(capped=sum(r['capped'] for r in results) if all('capped' in r for r in results) else None,
                tail95_seconds=times[-1] - times[int(.95 * (len(times) - 1))] if times else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--play', type=Path, default=PLAY, help='native nn_play binary (may be a frozen app snapshot)')
    p.add_argument('--initial-checkpoint', type=Path, help='warm-start a new run from compatible weights/optimizer/RNG; no old replay')
    p.add_argument('--balanced-replay', action='store_true', help='equal deck quotas instead of uniform decision-row sampling')
    p.add_argument('--iterations', type=int, default=10, help='updates; 0 = run until OUT/STOP exists')
    p.add_argument('--test-every', type=int, default=0, help='also evaluate every N updates; 0 disables')
    p.add_argument('--validation-share', type=float, default=0, help='evaluate when recorded validation wall share falls below this target; 0 disables')
    p.add_argument('--prune-replay', action='store_true', help='delete expired sample files and old TensorRT caches; retain models/results')
    p.add_argument('--max-train-rows', type=int, default=0, help='uniformly subsample loaded replay to this many rows; 0 = all')
    p.add_argument('--min-free-gib', type=float, default=0, help='clean stop before collection if disk reserve is low; 0 disables')
    p.add_argument('--games', type=int, default=1000)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--window', type=int, default=5, help='train on the last N iterations')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--sims', type=int, default=256)
    p.add_argument('--rollout-mix', type=float, default=0.5)
    p.add_argument('--test-sims', type=int, help='validation simulations (default --sims); 0 = the network alone')
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--gpu-batch', type=int, default=512)
    p.add_argument('--gpu-tokens', type=int, default=48)
    p.add_argument('--gpu-bank', default='', help='smaller fixed GPU batches, e.g. 256,128,32; same weights/tokens')
    p.add_argument('--gpu-wait-us', type=int, default=100, help='arrival-stall wait before a partial batch; 0 dispatches immediately')
    p.add_argument('--parallel', type=int, default=224, help='active game slots per CPU worker')
    p.add_argument('--timeout-ms', type=int, default=60000, help='Actor/request/server stall deadline')
    p.add_argument('--shutdown-timeout-ms', type=int, default=5000, help='Actor/server forced-exit grace after cancellation')
    p.add_argument('--test', type=Path, help='primary validation fights (JSON lines), greedy')
    p.add_argument('--test-in', type=Path, help='optional separate in-sample validation, not part of primary score')
    p.add_argument('--test-at', default='', help='iterations to test, e.g. 0,3,10')
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--fight-namespace', help='seed namespace; reuse a prior run name for matching training starts')
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume-from', type=int, help='resume a completed iteration; preserve replay and optimizer')
    p.add_argument('--export-reuse', action='store_true', help='run the ONNX exporter once per shape, then rewrite weights')
    a = p.parse_args()
    test_sims = a.sims if a.test_sims is None else a.test_sims
    if test_sims < 0:
        p.error('test-sims must be >= 0')
    try:
        gpu_bank = [int(b) for b in a.gpu_bank.split(',') if b]
    except ValueError:
        p.error('gpu-bank must be comma-separated integers')
    if len(set(gpu_bank)) != len(gpu_bank) or any(b <= 0 or b >= a.gpu_batch for b in gpu_bank):
        p.error('gpu-bank batches must be distinct, positive, and smaller than gpu-batch')
    if a.parallel < 1 or a.workers < 1 or a.gpu_batch < 1 or a.gpu_wait_us < 0 or a.timeout_ms <= 0 or a.shutdown_timeout_ms <= 0:
        p.error('positive workers/parallel/gpu-batch/timeouts; nonnegative GPU wait')
    if not 0 <= a.validation_share < 1:
        p.error('validation-share must be in [0,1)')
    if a.iterations < 0 or a.test_every < 0 or a.max_train_rows < 0 or a.window < 1:
        p.error('iterations/test-every/max-train-rows must be nonnegative; window must be positive')
    args = {k: str(v) for k, v in vars(a).items()}
    if a.resume_from is None:
        a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_loop(a, args, gpu_bank)


def run_loop(a, args, gpu_bank):
    if a.resume_from is not None:
        previous = json.loads((a.out / 'args.json').read_text())
        changed = {k for k in previous if previous[k] != args.get(k)}
        if changed - {'games', 'iterations', 'resume_from', 'test_every', 'prune_replay', 'max_train_rows', 'validation_share',
                      'export_reuse'}:
            raise ValueError(f'resume changed incompatible settings: {changed}')
        rows = [json.loads(line) for line in (a.out / 'log.jsonl').read_text().splitlines()]
        completed = [r['iteration'] for r in rows if 'iteration' in r]
        if not completed or a.resume_from != max(completed) or (a.iterations and a.resume_from >= a.iterations):
            raise ValueError('resume-from must be the latest completed iteration, below iterations (unless unlimited)')
        if (a.out / f'iter{a.resume_from + 1:03d}').exists():
            raise FileExistsError('archive the interrupted iteration before resuming; never append partial samples')
        checkpoint = torch.load(a.out / f'iter{a.resume_from:03d}' / 'model.pt', map_location='cpu', weights_only=True)
        if checkpoint.get('spec', dict(dim=64, layers=2, heads=2)) != dict(dim=a.dim, layers=a.layers, heads=a.heads):
            raise ValueError('checkpoint architecture does not match requested topology')
        (a.out / f'args-before-resume-{time.time_ns()}.json').write_text(json.dumps(previous, indent=1))
    (a.out / 'args.json').write_text(json.dumps(args, indent=1))
    run = Run(a.out, a.gpu_batch, gpu_bank, a.gpu_wait_us, a.parallel,
              a.timeout_ms, a.shutdown_timeout_ms, play_path=a.play)
    run.status(stage='initialization', phase='initialize')
    base = json.loads(a.start.read_text())
    if isinstance(base, list):
        fights(base, 'validate', a.games)
    tests = [json.loads(line) for line in a.test.read_text().splitlines() if line.strip()] if a.test else []
    tests_in = [json.loads(line) for line in a.test_in.read_text().splitlines() if line.strip()] if a.test_in else []
    test_at = {int(x) for x in a.test_at.split(',') if x}

    torch.manual_seed(0)
    net = Net(dim=a.dim, layers=a.layers, heads=a.heads).to(a.device)
    (a.out / 'model-spec.json').write_text(json.dumps(net.spec, indent=2))
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.initial_checkpoint and a.resume_from is None:
        warm = torch.load(a.initial_checkpoint, map_location='cpu', weights_only=True)
        if warm.get('spec') != net.spec:
            raise ValueError('initial checkpoint architecture differs from requested topology')
        net.load_state_dict(warm['net'])
        optimizer.load_state_dict(warm['optimizer'])
        torch.set_rng_state(warm['rng'])
        if warm.get('cuda_rng') and a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(warm['cuda_rng'])
        run.log(initial_checkpoint=str(a.initial_checkpoint), spec=net.spec)
    iteration_dir = lambda k: a.out / f'iter{k:03d}'
    exporter = Exporter() if a.export_reuse else None
    export_dynamic, export_fixed = (exporter.export, exporter.export_static) if exporter else (export, export_static)

    def fit(it, steps):
        samples = [s for k in range(max(0, it - a.window + 1), it + 1) for s in sorted(iteration_dir(k).glob('samples-*.bin'))]
        run.status(stage=f'iteration {it}', phase='load', files=len(samples))
        with Replay(samples, max_rows=a.max_train_rows, balanced=a.balanced_replay) as replay:
            expected = {f['fight_id'] for f in base} if isinstance(base, list) else set(replay.groups)
            if set(replay.groups) != expected:
                raise ValueError('replay does not contain every configured deck')
            run.status(stage=f'iteration {it}', phase='train', done=0, total=steps)
            train(net, optimizer, replay, None, None, steps, a.batch, a.device,
                  log=lambda s: (print(f'  {s}', flush=True), run.status(stage=f'iteration {it}', phase='train', detail=s)))
            (iteration_dir(it) / 'replay.json').write_text(json.dumps(dict(
                available=replay.available, selected=replay.selected, drawn=replay.drawn), indent=2))
            states = len(replay)
        torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'spec': net.spec,
                    'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []},
                   iteration_dir(it) / 'model.pt')
        run.status(stage=f'iteration {it}', phase='export')
        export_dynamic(net, iteration_dir(it) / 'model.onnx')
        export_fixed(net, iteration_dir(it) / 'model-gpu.onnx', a.gpu_batch, a.gpu_tokens)
        for b in gpu_bank:
            export_fixed(net, iteration_dir(it) / f'model-gpu-b{b}.onnx', b, a.gpu_tokens)
        net.to(a.device)
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()  # leave VRAM for the next native TensorRT worker
        return states

    def test(it):
        due = it in test_at or (a.test_every and it % a.test_every == 0)
        if a.validation_share and it not in test_at:
            history = [json.loads(s) for s in (a.out / 'log.jsonl').read_text().splitlines()]
            validation = sum(r['seconds'] for r in history if 'test' in r)
            learning = sum(r.get('play_seconds', 0) + r.get('train_seconds', 0) for r in history)
            due = validation < a.validation_share * (learning + validation)
        if not tests or not due: return
        t = time.time()
        results = run.play(iteration_dir(it) / 'model.onnx', tests, a.out / 'test' / f'iter{it:03d}', test_sims, 0.0,
                           a.rollout_mix, a.workers, stage=f'test iteration {it}')
        in_results = []
        if tests_in:
            in_results = run.play(iteration_dir(it) / 'model.onnx', tests_in, a.out / 'test' / f'iter{it:03d}' / 'in-sample',
                                  test_sims, 0.0, a.rollout_mix, a.workers, stage=f'test in-sample iteration {it}')
        run.log(test=it, games=len(results), wins=sum(r['won'] for r in results), seconds=round(time.time() - t),
                in_sample_games=len(in_results), in_sample_wins=sum(r['won'] for r in in_results), **collection_stats(results))
        if a.prune_replay:
            for sample in (a.out / 'test' / f'iter{it:03d}').rglob('samples-*.bin'):
                sample.unlink()

    if a.resume_from is not None:
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'rng' in checkpoint:
            torch.set_rng_state(checkpoint['rng'])
            if checkpoint.get('cuda_rng') and a.device.startswith('cuda'):
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        run.log(resume_from=a.resume_from, games_per_update=a.games, rng_restored='rng' in checkpoint)
        # A stop/crash between checkpoint commit and evaluation must not skip that evaluation.
        if not any(r.get('test') == a.resume_from for r in rows):
            if (partial_test := a.out / 'test' / f'iter{a.resume_from:03d}').exists():
                partial_test.rename(partial_test.with_name(f'.{partial_test.name}-interrupted-{time.time_ns()}'))
            test(a.resume_from)
        first = a.resume_from + 1
    else:
        # Iteration 0 is the initial (or warm-started) network, without old replay.
        iteration_dir(0).mkdir()
        t = time.time()
        export_dynamic(net, iteration_dir(0) / 'model.onnx')
        export_fixed(net, iteration_dir(0) / 'model-gpu.onnx', a.gpu_batch, a.gpu_tokens)
        for b in gpu_bank:
            export_fixed(net, iteration_dir(0) / f'model-gpu-b{b}.onnx', b, a.gpu_tokens)
        net.to(a.device); states = 0
        torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'spec': net.spec,
                    'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []},
                   iteration_dir(0) / 'model.pt')
        run.log(iteration=0, states=states, train_seconds=round(time.time() - t))
        test(0)
        first = 1

    for it in itertools.count(first):
        if (a.out / 'STOP').exists():
            run.status(stage='stopped', phase='stopped', last_completed=it - 1)
            return
        if a.iterations and it > a.iterations:
            break
        if a.min_free_gib:
            previous_bytes = sum(p.stat().st_size for p in iteration_dir(it - 1).glob('samples-*.bin'))
            reserve = a.min_free_gib * 1024**3 + max(3 * 1024**3, previous_bytes)
            if shutil.disk_usage(a.out).free < reserve:
                run.status(stage='stopped', phase='stopped', reason='disk reserve', last_completed=it - 1)
                run.log(stopped='disk reserve', required_free_bytes=reserve)
                return
        t0 = time.time()
        results = run.play(iteration_dir(it - 1) / 'model.onnx', fights(base, f'{a.fight_namespace or a.out.name}:{it}', a.games),
                           iteration_dir(it), a.sims, 1.0, a.rollout_mix, a.workers, stage=f'iteration {it}')
        t1 = time.time()
        states = fit(it, a.steps)
        run.log(iteration=it, games=len(results), wins=sum(r['won'] for r in results), states=states,
                play_seconds=round(t1 - t0), train_seconds=round(time.time() - t1), **collection_stats(results))
        test(it)
        if a.prune_replay:
            # Keep exactly the replay window; checkpoints/results remain for analysis.
            for old in a.out.glob('iter[0-9]*'):
                if old.is_dir() and old.name[4:].isdigit() and int(old.name[4:]) <= it - a.window:
                    for sample in old.glob('samples-*.bin'):
                        sample.unlink()
                        sample.with_suffix('.index.npz').unlink(missing_ok=True)
                    shutil.rmtree(old / 'tensorrt', ignore_errors=True)
    run.status(stage='done', phase='done')


if __name__ == '__main__':
    main()
