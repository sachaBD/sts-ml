"""Network-only reference on recorded states, including copies. Not gameplay throughput."""
import json
import statistics
import time

import numpy as np
import onnxruntime as ort

from throughput import MODEL, ROOT, fixed_model
from verify_throughput import states

OUT = ROOT / 'runs/nn-throughput-20261025'
x = states()
report = []
for batch in (32, 128, 256, 512, 1024):
    path = fixed_model(OUT, batch)
    providers = [('TensorrtExecutionProvider', {
        'trt_fp16_enable': True, 'trt_cuda_graph_enable': True,
        'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(path.parent / 'tensorrt'),
        'trt_timing_cache_enable': True, 'trt_timing_cache_path': str(ROOT / 'build/nn/tensorrt-timing'),
        'trt_builder_optimization_level': 3})]
    session = ort.InferenceSession(str(path), providers=providers)
    data = np.concatenate([x] * ((batch + len(x) - 1) // len(x)))[:batch].astype(np.float16)
    inp = ort.OrtValue.ortvalue_from_numpy(data, 'cuda', 0)
    value = ort.OrtValue.ortvalue_from_shape_and_type([batch], np.float32, 'cuda', 0)
    logits = ort.OrtValue.ortvalue_from_shape_and_type([batch, 48], np.float32, 'cuda', 0)
    binding = session.io_binding()
    binding.bind_ortvalue_input('tokens', inp)
    binding.bind_ortvalue_output('value', value)
    binding.bind_ortvalue_output('logits', logits)
    def step():
        inp.update_inplace(data)
        session.run_with_iobinding(binding)
        value.numpy(); logits.numpy()
    for _ in range(5): step()
    blocks = []
    for _ in range(3):
        begin = time.monotonic()
        for _ in range(50): step()
        blocks.append((time.monotonic() - begin) / 50)
    row = dict(batch=batch, block_seconds_per_call=blocks,
               median_evals_per_second=batch/statistics.median(blocks), blocks=3, calls_per_block=50)
    report.append(row)
    print(json.dumps(row), flush=True)
    del session
(OUT / 'network-reference.json').write_text(json.dumps(report, indent=2))
