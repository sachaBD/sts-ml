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
    def __init__(self, out, gpu_batch=1024, gpu_tokens=48):
        self.out = out
        self.gpu_batch, self.gpu_tokens = gpu_batch, gpu_tokens

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
        command = [str(PLAY), 'play', str(model), str(sample), '--threads', str(workers), '--parallel', '200',
                   '--gpu-model', str(model.with_name('model-gpu.onnx')), '--gpu-batch', str(self.gpu_batch),
                   '--gpu-tokens', str(self.gpu_tokens), '--sims', str(sims), '--gumbel-scale', str(gumbel_scale),
                   '--rollout-mix', str(rollout_mix)]
        # TensorRT and CUDA libraries come from the dependency-managed environment, not system-wide installs.
        packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        results = []
        self.status(stage=stage, phase='play', done=0, total=len(games))
        with starts.open() as source, (directory / 'results.jsonl').open('w') as journal, (directory / 'worker.log').open('w') as errors:
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
    p.add_argument('--test', type=Path, help='fights (JSON lines) to test on, greedy')
    p.add_argument('--test-at', default='', help='iterations to test, e.g. 0,3,10')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    run = Run(a.out, a.gpu_batch, a.gpu_tokens)
    base = json.loads(a.start.read_text())
    tests = [json.loads(line) for line in a.test.read_text().splitlines() if line.strip()] if a.test else []
    test_at = {int(x) for x in a.test_at.split(',') if x}

    torch.manual_seed(0)
    net = Net().to(a.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    iteration_dir = lambda k: a.out / f'iter{k:03d}'

    def fit(it, steps):
        samples = [s for k in range(max(0, it - a.window + 1), it + 1) for s in sorted(iteration_dir(k).glob('samples-*.bin'))]
        tokens, policy, won = read(samples)
        run.status(stage=f'iteration {it}', phase='train', done=0, total=steps)
        train(net, optimizer, tokens, policy, won, steps, a.batch, a.device,
              log=lambda s: (print(f'  {s}', flush=True), run.status(stage=f'iteration {it}', phase='train', detail=s)))
        torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, iteration_dir(it) / 'model.pt')
        export(net, iteration_dir(it) / 'model.onnx')
        export_static(net, iteration_dir(it) / 'model-gpu.onnx', a.gpu_batch, a.gpu_tokens)
        net.to(a.device)
        return len(won)

    def test(it):
        if it not in test_at or not tests: return
        t = time.time()
        results = run.play(iteration_dir(it) / 'model.onnx', tests, a.out / 'test' / f'iter{it:03d}', a.sims, 0.0,
                           a.rollout_mix, a.workers, stage=f'test iteration {it}')
        run.log(test=it, games=len(results), wins=sum(r['won'] for r in results), seconds=round(time.time() - t))

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
        net.to(a.device); states = 0
    run.log(iteration=0, states=states, train_seconds=round(time.time() - t))
    test(0)

    for it in range(1, a.iterations + 1):
        t0 = time.time()
        results = run.play(iteration_dir(it - 1) / 'model.onnx', fights(base, f'{a.out.name}:{it}', a.games),
                           iteration_dir(it), a.sims, 1.0, a.rollout_mix, a.workers, stage=f'iteration {it}')
        t1 = time.time()
        states = fit(it, a.steps)
        run.log(iteration=it, games=len(results), wins=sum(r['won'] for r in results), states=states,
                play_seconds=round(t1 - t0), train_seconds=round(time.time() - t1))
        test(it)
    run.status(stage='done', phase='done')


if __name__ == '__main__':
    main()
