"""Network evaluations/s on the GPU through ONNX Runtime: CUDA provider (fp32, CUDA graph) vs TensorRT (fp16, fused).
Fixed-shape model [1024, 48, WIDTH] exported from the panache iteration-10 network; timings include copies.
  LD_LIBRARY_PATH=<venv>/tensorrt_libs:<venv>/nvidia/*/lib PYTHONPATH=. .venv/bin/python experiments/25-10-26/a-gumbel/nn/bench_tensorrt.py
"""
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from agents.combat.nn.model import WIDTH, Net

RUN = 'runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel-panache-v1/iter010'
BATCH, TOKENS = 1024, 48
net = Net(); net.load_state_dict(torch.load(f'{RUN}/model.pt', weights_only=False)['net'])
out = Path('scratch/nn-tensorrt-benchmark')
out.mkdir(parents=True, exist_ok=True)
model = out / 'fp32-input.onnx'
# Same fp32 input for both providers: isolates internal precision/fusion from transport precision.
torch.onnx.export(net.eval(), (torch.zeros(BATCH, TOKENS, WIDTH),), str(model), dynamo=True,
                  input_names=['tokens'], output_names=['value', 'logits'])
x = np.zeros((BATCH, TOKENS, WIDTH), np.float32); x[:, 0, 0] = 1; x[:, 1:40, 0] = 2; x[:, 40:, 0] = 6
x[:, :, 1] = np.random.randint(1, 500, (BATCH, TOKENS)); x[:, :, 3:] = np.random.randn(BATCH, TOKENS, WIDTH - 3)

for name, providers in (
        ('CUDA fp32 + graph', [('CUDAExecutionProvider', {'enable_cuda_graph': '1'})]),
        ('TensorRT fp16 + graph', [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_cuda_graph_enable': True,
                                    'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(out / 'engines'),
                                    'trt_timing_cache_enable': True, 'trt_timing_cache_path': 'build/nn/tensorrt-timing',
                                    'trt_builder_optimization_level': 3})])):
    t0 = time.time()
    s = ort.InferenceSession(str(model), providers=providers)
    xi = ort.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
    v = ort.OrtValue.ortvalue_from_shape_and_type([BATCH], np.float32, 'cuda', 0)
    l = ort.OrtValue.ortvalue_from_shape_and_type([BATCH, TOKENS], np.float32, 'cuda', 0)
    io = s.io_binding(); io.bind_ortvalue_input('tokens', xi); io.bind_ortvalue_output('value', v); io.bind_ortvalue_output('logits', l)
    for _ in range(3): s.run_with_iobinding(io)
    setup = time.time() - t0
    calls, t = 50, time.time()
    for _ in range(calls):
        xi.update_inplace(x); s.run_with_iobinding(io); v.numpy(); l.numpy()
    dt = (time.time() - t) / calls
    t = time.time()
    for _ in range(calls): s.run_with_iobinding(io)
    run = (time.time() - t) / calls
    t = time.time()
    for _ in range(calls): xi.update_inplace(x)
    copy_in = (time.time() - t) / calls
    print(f'{name:24s} {dt * 1000:6.2f} ms/batch  {BATCH / dt:>10,.0f} evals/s   (network only {run * 1000:.2f} ms, '
          f'input copy {copy_in * 1000:.2f} ms; session + build {setup:.0f} s)', flush=True)
