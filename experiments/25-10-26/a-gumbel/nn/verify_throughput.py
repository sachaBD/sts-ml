"""Correctness checks for scheduler and batch-bank inference; no model training."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import onnxruntime as ort
import torch

from throughput import MODEL, ROOT, environment, fixed_model

OUT = ROOT / 'runs/nn-throughput-20261025'
VERIFY = OUT / 'verification'
VERIFY.mkdir(exist_ok=True)


def states(count=128):
    rows = []
    with (MODEL / 'samples-0.bin').open('rb') as source:
        while len(rows) < count:
            header = source.read(12)
            if not header:
                break
            tokens, actions, won = struct.unpack('iif', header)
            raw = source.read(tokens * 68 * 4)
            source.read(actions * 4)
            if tokens <= 48:
                rows.append(np.frombuffer(raw, np.float32).reshape(tokens, 68))
    x = np.zeros((len(rows), 48, 68), np.float32)
    for i, row in enumerate(rows):
        x[i, :len(row)] = row
    return x


def numerical(tokens=48, batches=(32, 128, 256, 512, 1024)):
    torch.set_num_threads(1)
    x = states()
    if tokens < 48:
        assert not np.any(x[:, tokens:, 0]), 'numerical test must not truncate real tokens'
        x = x[:, :tokens].copy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    cpu = ort.InferenceSession(str(MODEL / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    ref_value, ref_logits = cpu.run(None, {'tokens': x})
    rows = []
    for batch in batches:
        path = fixed_model(OUT, batch, tokens=tokens)
        providers = [('TensorrtExecutionProvider', {
            'trt_fp16_enable': True, 'trt_cuda_graph_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(path.parent / 'tensorrt'),
            'trt_timing_cache_enable': True, 'trt_timing_cache_path': str(ROOT / 'build/nn/tensorrt-timing'),
            'trt_builder_optimization_level': 3})]
        session = ort.InferenceSession(str(path), providers=providers)
        actual_value, actual_logits = [], []
        data = np.zeros((batch, tokens, 68), np.float16)
        data[:, 0, 0] = 1
        inp = ort.OrtValue.ortvalue_from_numpy(data, 'cuda', 0)
        value = ort.OrtValue.ortvalue_from_shape_and_type([batch], np.float32, 'cuda', 0)
        logits = ort.OrtValue.ortvalue_from_shape_and_type([batch, tokens], np.float32, 'cuda', 0)
        binding = session.io_binding()
        binding.bind_ortvalue_input('tokens', inp)
        binding.bind_ortvalue_output('value', value)
        binding.bind_ortvalue_output('logits', logits)
        for offset in range(0, len(x), batch):
            used = min(batch, len(x)-offset)
            data.fill(0)
            data[:, 0, 0] = 1
            data[:used] = x[offset:offset+used]
            inp.update_inplace(data)
            session.run_with_iobinding(binding)
            actual_value.extend(value.numpy()[:used])
            actual_logits.extend(logits.numpy()[:used])
        actual_value, actual_logits = np.array(actual_value), np.array(actual_logits)
        legal = x[..., 0] == 6
        assert np.isfinite(actual_value).all() and np.isfinite(actual_logits).all()
        np.testing.assert_allclose(actual_value, ref_value, atol=.003, rtol=.003)
        np.testing.assert_allclose(actual_logits[legal], ref_logits[legal], atol=.04, rtol=.01)
        row = dict(batch=batch, tokens=tokens, states=len(x), max_value_error=float(abs(actual_value-ref_value).max()),
                   max_legal_logit_error=float(abs(actual_logits[legal]-ref_logits[legal]).max()),
                   action_argmax_agreement=float((actual_logits.argmax(-1) == ref_logits.argmax(-1)).mean()))
        rows.append(row)
        print('NUMERICAL', json.dumps(row), flush=True)
        del session
    return rows


def scheduler():
    starts = '\n'.join((MODEL / 'starts.jsonl').read_text().splitlines()[:8]) + '\n'
    results = {}
    for name, binary, settings in [
        ('legacy', OUT / 'nn_play-before-investigation', []),
        ('ordered', ROOT / 'build/nn/agents/combat/nn/nn_play', []),
        ('ready2', ROOT / 'build/nn/agents/combat/nn/nn_play', ['--scheduler', 'ready']),
        ('ready4', ROOT / 'build/nn/agents/combat/nn/nn_play', ['--scheduler', 'ready', '--groups', '4']),
    ]:
        directory = VERIFY / name
        directory.mkdir(exist_ok=False)
        command = [str(binary), 'play', str(MODEL / 'model.onnx'), str(directory/'samples.bin'),
                   '--threads', '2', '--parallel', '8', '--sims', '4', '--gumbel-scale', '1', *settings]
        with (directory/'worker.log').open('w') as errors:
            result = subprocess.run(command, input=starts, text=True, capture_output=False,
                                    stdout=subprocess.PIPE, stderr=errors, env=environment(), check=True)
        (directory/'results.jsonl').write_text(result.stdout)
        canonical = {row['fight_id']: {k:v for k,v in row.items() if k != 'seconds'}
                     for row in map(json.loads, result.stdout.splitlines())}
        assert len(canonical) == 8
        results[name] = canonical
        assert canonical == results['legacy'], f'scheduler changes fight outcome: {name}'
        print('SCHEDULER', name, '8/8 identical outcomes, HP and decisions', flush=True)
    return dict(games=8, simulations=4, configurations=list(results), exact_outcome_parity=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numerical-only', action='store_true')
    parser.add_argument('--tokens', type=int, default=48)
    parser.add_argument('--batches', default='32,128,256,512,1024')
    args = parser.parse_args()
    report = dict(numerical=numerical(args.tokens, [int(b) for b in args.batches.split(',')]), time=time.time())
    if not args.numerical_only:
        report['scheduler'] = scheduler()
    filename = f'numerical-t{args.tokens}.json' if args.numerical_only else 'report.json'
    (VERIFY / filename).write_text(json.dumps(report, indent=2))
