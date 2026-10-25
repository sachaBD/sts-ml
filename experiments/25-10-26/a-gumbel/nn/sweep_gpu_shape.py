"""GPU inference throughput (network evaluations/s) by network shape and precision, batch 1024, CUDA graphs.

  PYTHONPATH=. .venv/bin/python experiments/25-10-26/a-gumbel/nn/sweep_gpu_shape.py
Random token contents (types/ids valid); throughput does not depend on the values.
"""
import time

import torch

from agents.combat.nn.model import ACTION, WIDTH, Net

BATCH = 1024


def tokens(batch, count):
    x = torch.zeros(batch, count, WIDTH)
    x[:, 0, 0] = 1
    x[:, 1:-8, 0] = 2
    x[:, -8:, 0] = ACTION
    x[:, :, 1] = torch.randint(1, 500, (batch, count))
    x[:, :, 3:] = torch.randn(batch, count, WIDTH - 3)
    return x.cuda()


def evals_per_second(dim, layers, count, dtype):
    net = Net(dim, layers, heads=max(1, dim // 32)).cuda().eval().to(dtype)
    x = tokens(BATCH, count).to(dtype)
    with torch.no_grad():
        for _ in range(3): net(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            net(x)
        torch.cuda.synchronize()
        calls, t = 50, time.time()
        for _ in range(calls): graph.replay()
        torch.cuda.synchronize()
    return BATCH * calls / (time.time() - t)


print(f'batch {BATCH}, CUDA graph replay; network evaluations per second')
print(f'{"width":>5} {"layers":>6} {"tokens":>6} {"fp32":>10} {"bf16":>10}')
for dim in (64, 128, 256):
    for layers in (2, 4):
        for count in (32, 48, 64):
            row = [evals_per_second(dim, layers, count, dt) for dt in (torch.float32, torch.bfloat16)]
            print(f'{dim:>5} {layers:>6} {count:>6} {row[0]:>10,.0f} {row[1]:>10,.0f}', flush=True)
