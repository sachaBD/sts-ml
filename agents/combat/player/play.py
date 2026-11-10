"""Plays fights with the native nn_play: one process per call, so each stage gets a fresh CUDA context
(a crash is recoverable by a new process) and never shares GPU state with the PyTorch learner."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys


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
    gpu_wait_us: int
    timeout_ms: int
    shutdown_timeout_ms: int


class DeviceSampler:
    """The GPU's utilisation and memory every 250 ms while a stage plays, as CSV (nvidia-smi)."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.stream = self.path.open('w')
        fields = 'timestamp,utilization.gpu,utilization.memory,memory.used,memory.total'
        self.process = subprocess.Popen(['nvidia-smi', '--id=0', f'--query-gpu={fields}', '--format=csv,nounits',
                                         '-lms', '250'], stdout=self.stream, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *error):
        self.process.terminate()
        self.process.wait()
        self.stream.close()


def play(binary, models, fights, directory, search, workers, *, progress):
    """Plays `fights` with nn_play `binary` and the exported model directory `models`; writes samples, results and
    logs to `directory`. Results are journaled as fights finish; progress(done, pid) follows them."""
    directory.mkdir(parents=True, exist_ok=True)
    samples = directory / 'samples-0.bin'
    if samples.exists():
        raise FileExistsError(f'{samples}: refusing to append duplicate gameplay')
    starts = directory / 'starts.jsonl'
    starts.write_text(''.join(json.dumps(f) + '\n' for f in fights))
    command = [str(binary), 'play', str(models), str(samples),
               '--threads', str(workers.actors), '--parallel', str(workers.slots),
               '--gpu-wait-us', str(workers.gpu_wait_us), '--timeout-ms', str(workers.timeout_ms),
               '--shutdown-timeout-ms', str(workers.shutdown_timeout_ms),
               '--telemetry', str(directory / 'telemetry.jsonl'),
               '--sims', str(search.sims), '--gumbel-scale', str(search.gumbel_scale),
               '--rollout-mix', str(search.rollout_mix), '--particles', str(search.particles)]
    # TensorRT and CUDA libraries come from the dependency-managed environment, not system-wide installs.
    packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    results = []
    with DeviceSampler(directory / 'gpu.csv'), starts.open() as source, \
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
