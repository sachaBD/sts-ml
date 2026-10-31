"""Plays fights with the native nn_play: one process per call, so each stage gets a fresh CUDA context
(a crash is recoverable by a new process) and never shares GPU state with the PyTorch learner."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from .metrics import DeviceSampler


@dataclass(frozen=True)
class Search:
    sims: int
    gumbel_scale: float  # 1 explores (self-play); 0 is deterministic (evaluation)
    rollout_mix: float
    particles: int


@dataclass(frozen=True)
class Workers:
    actors: int
    slots: int          # fights in flight per actor
    gpu_batch: int
    gpu_bank: tuple[int, ...]
    gpu_wait_us: int
    timeout_ms: int
    shutdown_timeout_ms: int


def model_files(directory, gpu_bank):
    """What nn_play loads from a model directory: the dynamic model, the fixed-batch GPU model and its smaller bank."""
    return dict(dynamic=directory / 'model.onnx', gpu=directory / 'model-gpu.onnx',
                bank=[directory / f'model-gpu-b{b}.onnx' for b in gpu_bank])


def play(binary, models, fights, directory, search, workers, progress=lambda done, pid: None, sample_devices=True):
    """Plays `fights` with nn_play `binary` and the models in `models`; writes samples, results and logs to `directory`.
    Results are journaled as fights finish; progress(done, pid) follows them."""
    directory.mkdir(parents=True, exist_ok=True)
    samples = directory / 'samples-0.bin'
    if samples.exists():
        raise FileExistsError(f'{samples}: refusing to append duplicate gameplay')
    starts = directory / 'starts.jsonl'
    starts.write_text(''.join(json.dumps(f) + '\n' for f in fights))
    files = model_files(models, workers.gpu_bank)
    command = [str(binary), 'play', str(files['dynamic']), str(samples),
               '--threads', str(workers.actors), '--parallel', str(workers.slots),
               '--gpu-model', str(files['gpu']), '--gpu-batch', str(workers.gpu_batch),
               '--gpu-wait-us', str(workers.gpu_wait_us), '--timeout-ms', str(workers.timeout_ms),
               '--shutdown-timeout-ms', str(workers.shutdown_timeout_ms),
               '--sims', str(search.sims), '--gumbel-scale', str(search.gumbel_scale),
               '--rollout-mix', str(search.rollout_mix), '--particles', str(search.particles)]
    if files['bank']:
        command += ['--gpu-bank', ','.join(map(str, files['bank']))]
    # TensorRT and CUDA libraries come from the dependency-managed environment, not system-wide installs.
    packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    results = []
    with DeviceSampler(directory / 'gpu.csv', enabled=sample_devices), starts.open() as source, \
            (directory / 'results.jsonl').open('w') as journal, (directory / 'worker.log').open('w') as errors:
        with subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=errors, text=True, env=env) as process:
            for line in process.stdout:
                results.append(json.loads(line)); journal.write(line); journal.flush()
                progress(len(results), process.pid)
        if process.returncode:
            raise RuntimeError(f'nn_play exited {process.returncode}; see {directory / "worker.log"}')
    if len(results) != len(fights):
        raise RuntimeError(f'nn_play returned {len(results)} of {len(fights)} fights')
    return results
