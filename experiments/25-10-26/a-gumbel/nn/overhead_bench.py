"""CPU benchmark of per-update replay load and export: read vs Replay, full export vs Exporter.

  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python experiments/25-10-26/a-gumbel/nn/overhead_bench.py OUT

Synthetic files of 20k decisions (the panache run has ~50k), 5-file window; 5 updates after a cold one.
"""
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from agents.combat.nn.data import Replay, read  # noqa: E402
from agents.combat.nn.graphs import Exporter  # noqa: E402
from agents.combat.nn.model import Net, export, export_static  # noqa: E402
from agents.combat.nn.test_data import write  # noqa: E402


def timed(f, *args):
    start = time.perf_counter()
    result = f(*args)
    return time.perf_counter() - start, result


out = Path(sys.argv[1])
out.mkdir(parents=True)
torch.set_num_threads(2)
files = [write(out / f'samples-{k}.bin', k, 20000, 39) for k in range(10)]
replay, rows = Replay(), []
for update in range(6):
    window = files[update:update + 5]
    old, a = timed(read, window)
    new, b = timed(replay.read, window)
    rows.append(dict(update=update, read=old, replay=new, same=all(torch.equal(x, y) for x, y in zip(a, b))))
    del a, b  # one dense replay at a time
    print(json.dumps(rows[-1]), flush=True)

net, exporter = Net(dim=128, layers=4, heads=4), Exporter()
optimizer = torch.optim.AdamW(net.parameters())
for update in range(6):
    x = torch.randn(4, 6, 68)  # one real optimizer step: context token, then action tokens
    x[..., 0], x[:, 0, 0], x[..., 1:3] = 6, 1, torch.randint(0, 50, (4, 6, 2))
    value, logits = net(x)
    (value.sum() + logits.log_softmax(-1)[:, 1:].sum()).backward()
    optimizer.step(), optimizer.zero_grad()
    full, fast = out / f'full{update}', out / f'reuse{update}'
    full.mkdir(), fast.mkdir()
    start = time.perf_counter()
    export(net, full / 'model.onnx')
    for batch in (512, 256, 32):
        export_static(net, full / f'model-gpu-b{batch}.onnx', batch, 48)
    middle = time.perf_counter()
    exporter.export(net, fast / 'model.onnx')
    for batch in (512, 256, 32):
        exporter.export_static(net, fast / f'model-gpu-b{batch}.onnx', batch, 48)
    end = time.perf_counter()
    same = all((full / p.name).read_bytes() == p.read_bytes() for p in fast.iterdir())
    rows.append(dict(update=update, export=middle - start, reuse=end - middle, same=same))
    print(json.dumps(rows[-1]), flush=True)
(out / 'results.json').write_text(json.dumps(rows, indent=1))
