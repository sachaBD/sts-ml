"""Gumbel self-play loop: [bootstrap on recorded games] -> (play -> train on recent games -> export) x N, with tests.

  python -m agents.combat.nn.loop --start START.json --out DIR [--bootstrap GAMES.jsonl] [--test FIGHTS.jsonl --test-at 0,3,10]

START.json: one fight {"fight_id", "start"} (combat_v4 start); training fights reuse it with new seeds.
GAMES.jsonl: recorded games for `nn_play import` (e.g. MCTS); trained on as iteration 0's samples.
Writes DIR/iterNNN/{samples-*.bin, results.jsonl, model.pt, model.onnx}, DIR/test/iterNNN/results.jsonl,
DIR/log.jsonl (one line per iteration or test) and DIR/status.json (live stage and progress).
"""
import argparse
import copy
import hashlib
import json
import subprocess
import os
import sys
import time
from pathlib import Path

import torch

from .data import read
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
    out = []
    for i in range(n):
        seed = int.from_bytes(hashlib.sha256(f'{name}:{i}'.encode()).digest()[:8], 'little')
        f = copy.deepcopy(base)
        f['fight_id'] = f'{name}:{i}'
        f['start'].update(seed=seed, misc_rng=rng_state(seed), potion_rng=rng_state(seed))
        out.append(f)
    return out


class Run:
    def __init__(self, out, gpu_batch=1024, gpu_tokens=48, gpu_bank=(), gpu_wait_us=1000,
                 parallel=200, groups=2, scheduler='ordered', timeout_ms=60000, shutdown_timeout_ms=5000,
                 sample_devices=True):
        self.out = out
        self.gpu_batch, self.gpu_tokens = gpu_batch, gpu_tokens
        self.gpu_bank, self.gpu_wait_us = gpu_bank, gpu_wait_us
        self.parallel, self.groups, self.scheduler = parallel, groups, scheduler
        self.timeout_ms, self.shutdown_timeout_ms = timeout_ms, shutdown_timeout_ms
        self.sample_devices = sample_devices

    def status(self, **fields):
        tmp = self.out / '.status.json'
        tmp.write_text(json.dumps(dict(time=time.time(), **fields)))
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
        command = [str(PLAY), 'play', str(model), str(sample), '--threads', str(workers), '--parallel', str(self.parallel),
                   '--gpu-model', str(model.with_name('model-gpu.onnx')), '--gpu-batch', str(self.gpu_batch),
                   '--gpu-tokens', str(self.gpu_tokens), '--sims', str(sims), '--gumbel-scale', str(gumbel_scale),
                   '--rollout-mix', str(rollout_mix), '--gpu-wait-us', str(self.gpu_wait_us),
                   '--groups', str(self.groups), '--scheduler', self.scheduler,
                   '--timeout-ms', str(self.timeout_ms), '--shutdown-timeout-ms', str(self.shutdown_timeout_ms)]
        if self.gpu_bank:
            command += ['--gpu-bank', ','.join(str(model.with_name(f'model-gpu-b{b}.onnx')) for b in self.gpu_bank)]
        # TensorRT and CUDA libraries come from the dependency-managed environment, not system-wide installs.
        packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        results = []
        self.status(stage=stage, phase='play', done=0, total=len(games))
        with DeviceSampler(directory / 'gpu.csv', enabled=self.sample_devices), \
                starts.open() as source, (directory / 'results.jsonl').open('w') as journal, (directory / 'worker.log').open('w') as errors:
            with subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=errors, text=True, env=env) as process:
                for line in process.stdout:
                    results.append(json.loads(line)); journal.write(line); journal.flush()
                    self.status(stage=stage, phase='play', done=len(results), total=len(games))
            if process.returncode:
                raise RuntimeError(f'nn_play exited {process.returncode}; see {directory / "worker.log"}')
        if len(results) != len(games):
            raise RuntimeError(f'nn_play returned {len(results)} of {len(games)} games')
        return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=Path, help='recorded games to import and train on first')
    p.add_argument('--bootstrap-steps', type=int, default=2000)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--games', type=int, default=1000)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--window', type=int, default=5, help='train on the last N iterations (iteration 0 = bootstrap games)')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--sims', type=int, default=256)
    p.add_argument('--rollout-mix', type=float, default=0.5)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--gpu-batch', type=int, default=1024)
    p.add_argument('--gpu-tokens', type=int, default=48)
    p.add_argument('--gpu-bank', default='', help='smaller fixed GPU batches, e.g. 256,128,32; same weights/tokens')
    p.add_argument('--gpu-wait-us', type=int, default=1000, help='arrival-stall wait before a partial batch; 0 dispatches immediately')
    p.add_argument('--parallel', type=int, default=200, help='active game slots per CPU worker')
    p.add_argument('--groups', type=int, default=2, help='independent inference groups per worker')
    p.add_argument('--scheduler', choices=['local', 'ordered', 'ready'], default='ordered',
                   help='local = worker-owned games + coordinator; threads count simulation workers')
    p.add_argument('--timeout-ms', type=int, default=60000, help='local scheduler request/worker stall deadline')
    p.add_argument('--shutdown-timeout-ms', type=int, default=5000, help='local scheduler forced-exit grace after cancellation')
    p.add_argument('--test', type=Path, help='fights (JSON lines) to test on, greedy')
    p.add_argument('--test-at', default='', help='iterations to test, e.g. 0,3,10')
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--fight-namespace', help='seed namespace; reuse a prior run name for matching training starts')
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume-from', type=int, help='resume a completed iteration; preserve replay and optimizer')
    a = p.parse_args()
    try:
        gpu_bank = [int(b) for b in a.gpu_bank.split(',') if b]
    except ValueError:
        p.error('gpu-bank must be comma-separated integers')
    if len(set(gpu_bank)) != len(gpu_bank) or any(b <= 0 or b >= a.gpu_batch for b in gpu_bank):
        p.error('gpu-bank batches must be distinct, positive, and smaller than gpu-batch')
    if a.parallel < 1 or a.workers < 1 or a.gpu_batch < 1 or a.gpu_wait_us < 0 or a.timeout_ms <= 0 or a.shutdown_timeout_ms <= 0:
        p.error('positive workers/parallel/gpu-batch/timeouts; nonnegative GPU wait')
    if a.scheduler != 'local' and (a.groups < 2 or a.parallel < a.groups or (a.parallel + a.groups - 1) // a.groups > a.gpu_batch):
        p.error('legacy scheduler requires parallel >= groups >= 2; each group must fit gpu-batch')
    args = {k: str(v) for k, v in vars(a).items()}
    if a.resume_from is None:
        a.out.mkdir(parents=True, exist_ok=False)
    else:
        previous = json.loads((a.out / 'args.json').read_text())
        changed = {k for k in previous if previous[k] != args[k]}
        legacy_defaults = dict(gpu_bank='', gpu_wait_us=1000, parallel=200, groups=2, scheduler='ordered',
                               timeout_ms=60000, shutdown_timeout_ms=5000)
        changed |= {k for k, default in legacy_defaults.items() if previous.get(k, str(default)) != args[k]}
        if changed - {'games', 'iterations', 'resume_from'}:
            raise ValueError(f'resume changed incompatible settings: {changed}')
        rows = [json.loads(line) for line in (a.out / 'log.jsonl').read_text().splitlines()]
        completed = [r['iteration'] for r in rows if 'iteration' in r]
        if not completed or a.resume_from != max(completed) or a.resume_from >= a.iterations:
            raise ValueError('resume-from must be the latest completed iteration, below iterations')
        if (a.out / f'iter{a.resume_from + 1:03d}').exists():
            raise FileExistsError('archive the interrupted iteration before resuming; never append partial samples')
        checkpoint = torch.load(a.out / f'iter{a.resume_from:03d}' / 'model.pt', map_location='cpu', weights_only=True)
        if checkpoint.get('spec', dict(dim=64, layers=2, heads=2)) != dict(dim=a.dim, layers=a.layers, heads=a.heads):
            raise ValueError('checkpoint architecture does not match requested topology')
        (a.out / f'args-before-resume-{time.time_ns()}.json').write_text(json.dumps(previous, indent=1))
    (a.out / 'args.json').write_text(json.dumps(args, indent=1))
    run = Run(a.out, a.gpu_batch, a.gpu_tokens, gpu_bank, a.gpu_wait_us, a.parallel, a.groups, a.scheduler,
              a.timeout_ms, a.shutdown_timeout_ms)
    base = json.loads(a.start.read_text())
    tests = [json.loads(line) for line in a.test.read_text().splitlines() if line.strip()] if a.test else []
    test_at = {int(x) for x in a.test_at.split(',') if x}

    torch.manual_seed(0)
    net = Net(dim=a.dim, layers=a.layers, heads=a.heads).to(a.device)
    (a.out / 'model-spec.json').write_text(json.dumps(net.spec, indent=2))
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    iteration_dir = lambda k: a.out / f'iter{k:03d}'

    def fit(it, steps):
        samples = [s for k in range(max(0, it - a.window + 1), it + 1) for s in sorted(iteration_dir(k).glob('samples-*.bin'))]
        tokens, policy, won = read(samples)
        run.status(stage=f'iteration {it}', phase='train', done=0, total=steps)
        train(net, optimizer, tokens, policy, won, steps, a.batch, a.device,
              log=lambda s: (print(f'  {s}', flush=True), run.status(stage=f'iteration {it}', phase='train', detail=s)))
        torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'spec': net.spec,
                    'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []},
                   iteration_dir(it) / 'model.pt')
        export(net, iteration_dir(it) / 'model.onnx')
        export_static(net, iteration_dir(it) / 'model-gpu.onnx', a.gpu_batch, a.gpu_tokens)
        for b in gpu_bank:
            export_static(net, iteration_dir(it) / f'model-gpu-b{b}.onnx', b, a.gpu_tokens)
        net.to(a.device)
        return len(won)

    def test(it):
        if it not in test_at or not tests: return
        t = time.time()
        results = run.play(iteration_dir(it) / 'model.onnx', tests, a.out / 'test' / f'iter{it:03d}', a.sims, 0.0,
                           a.rollout_mix, a.workers, stage=f'test iteration {it}')
        run.log(test=it, games=len(results), wins=sum(r['won'] for r in results), seconds=round(time.time() - t))

    if a.resume_from is not None:
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'rng' in checkpoint:
            torch.set_rng_state(checkpoint['rng'])
            if checkpoint.get('cuda_rng') and a.device.startswith('cuda'):
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        run.log(resume_from=a.resume_from, games_per_update=a.games, rng_restored='rng' in checkpoint)
        first = a.resume_from + 1
    else:
        # Iteration 0: the bootstrap games (or an untrained network).
        iteration_dir(0).mkdir()
        t = time.time()
        if a.bootstrap:
            run.status(stage='iteration 0', phase='import')
            with a.bootstrap.open() as games:
                subprocess.run([str(PLAY), 'import', str(iteration_dir(0) / 'samples-bootstrap.bin')], stdin=games, check=True)
            states = fit(0, a.bootstrap_steps)
        else:
            export(net, iteration_dir(0) / 'model.onnx')
            export_static(net, iteration_dir(0) / 'model-gpu.onnx', a.gpu_batch, a.gpu_tokens)
            for b in gpu_bank:
                export_static(net, iteration_dir(0) / f'model-gpu-b{b}.onnx', b, a.gpu_tokens)
            net.to(a.device); states = 0
        run.log(iteration=0, states=states, train_seconds=round(time.time() - t))
        test(0)
        first = 1

    for it in range(first, a.iterations + 1):
        t0 = time.time()
        results = run.play(iteration_dir(it - 1) / 'model.onnx', fights(base, f'{a.fight_namespace or a.out.name}:{it}', a.games),
                           iteration_dir(it), a.sims, 1.0, a.rollout_mix, a.workers, stage=f'iteration {it}')
        t1 = time.time()
        states = fit(it, a.steps)
        run.log(iteration=it, games=len(results), wins=sum(r['won'] for r in results), states=states,
                play_seconds=round(t1 - t0), train_seconds=round(time.time() - t1))
        test(it)
    run.status(stage='done', phase='done')


if __name__ == '__main__':
    main()
