"""Matched frozen-model full-game throughput benchmark; no training.

Generated outputs only under --out. Completed trials are skipped; partial trials must be archived.
Example: PYTHONPATH=. .venv/bin/python experiments/25-10-26/a-gumbel/nn/throughput.py --batches 1024,512,256
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / 'runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel64-panache-128x4-v1/iter010'


def environment():
    packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
    return dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))


def fixed_model(out, batch, model=MODEL, tokens=48):
    if batch == 1024 and tokens == 48:
        return model / 'model-gpu.onnx'
    path = out / 'models' / (f'b{batch}' if tokens == 48 else f'b{batch}-t{tokens}') / 'model-gpu.onnx'
    if path.exists():
        return path
    import torch
    from agents.combat.nn.model import Net, export_static
    torch.set_num_threads(1)
    checkpoint = torch.load(model / 'model.pt', map_location='cpu', weights_only=True)
    net = Net(**checkpoint['spec']).eval()
    net.load_state_dict(checkpoint['net'])
    path.parent.mkdir(parents=True, exist_ok=True)
    export_static(net, path, batch, tokens)
    return path


def extract(pattern, text):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f'missing benchmark metric: {pattern}')
    return float(match.group(1))


def trial(args, batch, repeat):
    name = f'{args.tag}-b{batch}-p{args.parallel}-g{args.groups}-{args.scheduler}-n{args.games}-r{repeat}'
    if args.tokens != 48:
        name += f'-t{args.tokens}'
    if args.adaptive:
        name += '-a' + args.adaptive.replace(',', '_')
    if args.wait_us != 1000:
        name += f'-w{args.wait_us}'
    directory = args.out / 'trials' / name
    summary = directory / 'summary.json'
    if summary.exists():
        result = json.loads(summary.read_text())
        recorded = json.loads((directory / 'command.json').read_text())
        if recorded['binary_sha256'] != hashlib.sha256(args.binary.resolve().read_bytes()).hexdigest():
            raise FileExistsError(f'{directory}: completed with another binary; use a new tag')
        if recorded['model_sha256'] != hashlib.sha256((MODEL / 'model.pt').read_bytes()).hexdigest():
            raise FileExistsError(f'{directory}: frozen model changed; use a new artifact root')
        print('completed, skipping:', name, flush=True)
        return result
    if directory.exists():
        raise FileExistsError(f'{directory}: interrupted trial; archive it before retrying')
    prepare_begin = time.monotonic()
    fixed = fixed_model(args.out, batch, tokens=args.tokens)
    # Optional adaptive bank (the first model is the maximum shape).
    bank = [fixed_model(args.out, int(b), tokens=args.tokens) for b in args.adaptive.split(',') if b] if args.adaptive else []
    preparation = time.monotonic() - prepare_begin
    directory.mkdir(parents=True)
    starts = MODEL / 'starts.jsonl'
    rows = starts.read_text().splitlines()
    if args.games != len(rows):
        # Larger jobs deterministically extend the same prefix with fresh starts, never repeat identical games.
        from agents.combat.nn.loop import fights
        base = json.loads(rows[0])
        if args.games > len(rows):
            rows += [json.dumps(f) for f in fights(base, 'throughput-reserve', args.games - len(rows))]
        else:
            rows = rows[:args.games]
    input_file = directory / 'starts.jsonl'
    input_file.write_text('\n'.join(rows) + '\n')
    binary = args.binary.resolve()
    command = [str(binary), 'play', str(MODEL / 'model.onnx'), str(directory / 'samples.bin'),
               '--threads', str(args.workers), '--parallel', str(args.parallel),
               '--gpu-model', str(fixed), '--gpu-batch', str(batch), '--gpu-tokens', str(args.tokens),
               '--sims', '64', '--gumbel-scale', '1', '--rollout-mix', '.5']
    if args.scheduler != 'ordered' or args.groups != 2:
        command += ['--scheduler', args.scheduler, '--groups', str(args.groups)]
    if bank:
        command += ['--gpu-bank', ','.join(map(str, bank))]
    if args.wait_us != 1000:
        command += ['--gpu-wait-us', str(args.wait_us)]
    metadata = dict(command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    start_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256((MODEL / 'model.pt').read_bytes()).hexdigest(),
                    time=time.time(), config=vars(args) | dict(out=str(args.out), binary=str(binary)))
    (directory / 'command.json').write_text(json.dumps(metadata, default=str, indent=2))
    print('START', name, flush=True)
    completions = []
    fields = 'timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu'
    with (directory / 'gpu.csv').open('w') as telemetry, input_file.open() as source, \
            (directory / 'worker.log').open('w') as errors, (directory / 'results.jsonl').open('w') as journal, \
            (directory / 'progress.jsonl').open('w') as progress:
        sampler = subprocess.Popen(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,nounits', '-lms', '500'],
                                   stdout=telemetry, stderr=subprocess.DEVNULL)
        begin, wall_begin = time.monotonic(), time.time()
        try:
            timed = ['/usr/bin/time', '-v', '-o', str(directory / 'time.txt'), *command]
            with subprocess.Popen(timed, stdin=source, stdout=subprocess.PIPE, stderr=errors, text=True,
                                  env=environment()) as process:
                for line in process.stdout:
                    row = json.loads(line)
                    journal.write(line); journal.flush()
                    elapsed = time.monotonic() - begin
                    completions.append(dict(elapsed=elapsed, fight_id=row['fight_id'], won=row['won'],
                                            hp=row['hp'], decisions=row['decisions']))
                    progress.write(json.dumps(dict(done=len(completions), elapsed=elapsed)) + '\n'); progress.flush()
                    if len(completions) % 500 == 0:
                        print(name, len(completions), '/', len(rows), f'{elapsed:.1f}s', flush=True)
            if process.returncode:
                raise RuntimeError(f'worker exit {process.returncode}: {directory / "worker.log"}')
        finally:
            sampler.terminate()
            sampler.wait()
        elapsed = time.monotonic() - begin
    if len(completions) != len(rows) or len({r['fight_id'] for r in completions}) != len(rows):
        raise RuntimeError('missing or duplicate fight results')
    log = (directory / 'worker.log').read_text()
    gameplay = extract(r'timing: gameplay ([\d.]+)', log)
    setup = extract(r'timing: evaluator setup ([\d.]+)', log)
    batches = extract(r'gpu: (\d+) batches', log)
    mean = extract(r'batches, mean ([\d.]+) leaves', log)
    busy = extract(r'leaves, ([\d.]+) s busy', log)
    exact = re.findall(r'gpu_stats: (\{[^\n]+\})', log)
    stats = json.loads(exact[-1]) if exact else {}
    leaves = stats.get('leaves', batches * mean)
    capacity = stats.get('capacity', batches * batch)
    timing = (directory / 'time.txt').read_text()
    cpu = extract(r'User time \(seconds\): ([\d.]+)', timing) + extract(r'System time \(seconds\): ([\d.]+)', timing)
    milestones = {str(percent): completions[max(0, (len(rows) * percent + 99) // 100 - 1)]['elapsed']
                  for percent in (10, 25, 50, 75, 90, 95, 99, 100)}
    end = completions[-1]['elapsed']
    tail = {str(percent): end - milestones[str(percent)] for percent in (50, 75, 90, 95, 99)}
    result = dict(name=name, batch=batch, tokens=args.tokens, groups=args.groups, scheduler=args.scheduler, parallel=args.parallel,
                  adaptive=args.adaptive, wait_us=args.wait_us, games=len(rows), repeat=repeat, wins=sum(r['won'] for r in completions),
                  elapsed_seconds=elapsed, model_preparation_seconds=preparation,
                  setup_seconds=setup, gameplay_seconds=gameplay,
                  games_per_hour=len(rows)*3600/gameplay, leaves=leaves, useful_evals_per_second=leaves/gameplay,
                  total_evals_per_second=(leaves + stats.get('cpu_leaves', 0))/gameplay,
                  batch_fill=leaves/capacity, evaluator_busy_fraction=busy/gameplay,
                  cpu_logical_cores=cpu/elapsed, milestones=milestones, tail_seconds=tail,
                  gpu_stats=stats, wall_begin=wall_begin)
    temporary = summary.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)); temporary.replace(summary)
    print('RESULT', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/nn-throughput-20261025')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/nn/agents/combat/nn/nn_play')
    parser.add_argument('--batches', default='1024,512,256')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--repeat-offset', type=int, default=0, help='number trials for externally interleaved configurations')
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--tokens', type=int, default=48)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--parallel', type=int, default=200)
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--scheduler', choices=['local', 'ordered', 'ready'], default='ordered')
    parser.add_argument('--adaptive', default='', help='comma-separated smaller batch shapes')
    parser.add_argument('--wait-us', type=int, default=1000)
    parser.add_argument('--tag', default='screen')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    batches = [int(x) for x in args.batches.split(',')]
    results = []
    for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
        order = batches if repeat % 2 == 0 else list(reversed(batches))
        for batch in order:
            results.append(trial(args, batch, repeat))
    (args.out / f'{args.tag}-report.json').write_text(json.dumps(results, indent=2))
    print('ALL COMPLETE:', args.out / f'{args.tag}-report.json', flush=True)


if __name__ == '__main__':
    main()
