"""Check shared TensorRT timing cache and fixed-batch throughput.

No gameplay/training. Each engine directory is unique; the timing cache is shared with nn_play.
Usage: PYTHONPATH=. .venv/bin/python experiments/25-10-26/a-gumbel/nn/verify_gpu.py
"""
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import onnxruntime as ort
import torch

from agents.combat.nn.data import read
from agents.combat.nn.model import Net, WIDTH, export_static

ROOT = Path('scratch/nn-gpu-verification') / time.strftime('%Y%m%d-%H%M%S')
ROOT.mkdir(parents=True)
MODEL = Path('runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel-panache-v1/iter010')
BATCH, TOKENS = 1024, 48
net = Net().eval()
net.load_state_dict(torch.load(MODEL / 'model.pt', map_location='cpu', weights_only=False)['net'])
export_static(net, ROOT / 'model.onnx', BATCH, TOKENS)
data, _, _ = read([MODEL / 'samples-0.bin'])
x = np.zeros((BATCH, TOKENS, WIDTH), np.float32)
x[:, :data.shape[1]] = data[:BATCH].float().numpy()
half = x.astype(np.float16)

report = []
for k in range(2):
    opts = {'trt_fp16_enable': True, 'trt_cuda_graph_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(ROOT / f'engine-{k}'),
            'trt_timing_cache_enable': True, 'trt_timing_cache_path': os.path.abspath('build/nn/tensorrt-timing'),
            'trt_builder_optimization_level': 3}
    t = time.monotonic()
    session = ort.InferenceSession(str(ROOT / 'model.onnx'), providers=[('TensorrtExecutionProvider', opts)])
    inp = ort.OrtValue.ortvalue_from_numpy(half, 'cuda', 0)
    value = ort.OrtValue.ortvalue_from_shape_and_type([BATCH], np.float32, 'cuda', 0)
    logits = ort.OrtValue.ortvalue_from_shape_and_type([BATCH, TOKENS], np.float32, 'cuda', 0)
    io = session.io_binding()
    io.bind_ortvalue_input('tokens', inp)
    io.bind_ortvalue_output('value', value)
    io.bind_ortvalue_output('logits', logits)
    for _ in range(3): session.run_with_iobinding(io)
    build = time.monotonic() - t
    blocks = []
    for _ in range(3):
        t = time.monotonic()
        for _ in range(20):
            inp.update_inplace(half)
            session.run_with_iobinding(io)
            value.numpy(); logits.numpy()
        blocks.append((time.monotonic() - t) / 20 * 1000)
    row = dict(build=k, setup_seconds=round(build, 2), batch_ms=blocks,
               evals_per_second=round(BATCH / (statistics.median(blocks) / 1000)))
    report.append(row)
    print(json.dumps(row), flush=True)
(ROOT / 'report.json').write_text(json.dumps(report, indent=2))
print('report:', ROOT / 'report.json', flush=True)
