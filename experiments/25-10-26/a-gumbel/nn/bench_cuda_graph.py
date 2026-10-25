"""ms per network call on the GPU: plain CUDA provider vs CUDA graphs (fixed shapes, IO binding).

Each timed call includes copying a new input batch to the GPU and the outputs back, as the search would.
  PYTHONPATH=. .venv/bin/python experiments/25-10-26/a-gumbel/nn/bench_cuda_graph.py
"""
import glob
import time

import numpy as np
import onnxruntime as ort
import torch

from agents.combat.nn.data import read
from agents.combat.nn.model import WIDTH, Net

RUN = 'runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel-panache-v1/iter010'
TOKENS = 48  # fixed token count (pad up); the recorded states have at most 39

tokens, _, _ = read(sorted(glob.glob(f'{RUN}/samples-*.bin')))
pool = np.zeros((4096, TOKENS, tokens.shape[2]), np.float32)
pool[:, :tokens.shape[1]] = tokens[:4096].float().numpy()


def plain(batch, calls):
    s = ort.InferenceSession(f'{RUN}/model.onnx', providers=['CUDAExecutionProvider'])
    x = pool[:batch]
    for _ in range(3): s.run(None, {'tokens': x})
    t = time.time()
    for _ in range(calls): s.run(None, {'tokens': x})
    return (time.time() - t) / calls


def static_model(batch):
    """The same network exported with fixed shapes: no dynamic shape arithmetic left on the CPU."""
    net = Net()
    net.load_state_dict(torch.load(f'{RUN}/model.pt', weights_only=False)['net'])
    path = f'/tmp/static-{batch}.onnx'
    torch.onnx.export(net.eval(), (torch.zeros(batch, TOKENS, WIDTH),), path, dynamo=True,
                      input_names=['tokens'], output_names=['value', 'logits'])
    return path


def graph(batch, calls):
    s = ort.InferenceSession(static_model(batch), providers=[('CUDAExecutionProvider', {'enable_cuda_graph': '1'})])
    x = ort.OrtValue.ortvalue_from_numpy(pool[:batch], 'cuda', 0)
    value = ort.OrtValue.ortvalue_from_shape_and_type([batch], np.float32, 'cuda', 0)
    logits = ort.OrtValue.ortvalue_from_shape_and_type([batch, TOKENS], np.float32, 'cuda', 0)
    io = s.io_binding()
    io.bind_ortvalue_input('tokens', x)
    io.bind_ortvalue_output('value', value)
    io.bind_ortvalue_output('logits', logits)
    for _ in range(3): s.run_with_iobinding(io)  # the first run captures the graph
    t = time.time()
    for _ in range(calls):
        x.update_inplace(pool[:batch])           # new inputs into the captured buffer
        s.run_with_iobinding(io)
        value.numpy(); logits.numpy()             # outputs back to the CPU
    return (time.time() - t) / calls


print(f'tokens per state fixed at {TOKENS}; times include input and output copies')
print(f'{"batch":>6} {"plain ms":>9} {"graph ms":>9} {"plain evals/s":>14} {"graph evals/s":>14}')
for batch in (1, 16, 64, 256, 1024, 4096):
    calls = max(5, 1000 // batch)
    p, g = plain(batch, calls), graph(batch, calls)
    print(f'{batch:>6} {p * 1000:>9.2f} {g * 1000:>9.2f} {batch / p:>14,.0f} {batch / g:>14,.0f}', flush=True)
