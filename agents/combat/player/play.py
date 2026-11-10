"""Plays fights with the native nn_play: one process per call, so each stage gets a fresh CUDA context
(a crash is recoverable by a new process) and never shares GPU state with the PyTorch learner."""
from dataclasses import dataclass
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

from agents.combat.nn.training.export import model_files


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
    gpu_wide_batch: int = 32


class DeviceSampler:
    """Optional per-stage NVML CSV. Missing nvidia-smi leaves GPU counters unavailable."""
    def __init__(self, path, enabled=True):
        self.path, self.enabled = Path(path), enabled
        self.process = self.stream = None

    def __enter__(self):
        binary = shutil.which('nvidia-smi') if self.enabled else None
        if binary:
            self.stream = self.path.open('w')
            fields = 'timestamp,utilization.gpu,utilization.memory,memory.used,memory.total'
            try:
                self.process = subprocess.Popen([binary,'--id=0',f'--query-gpu={fields}','--format=csv,nounits','-lms','250'],
                                                stdout=self.stream,stderr=subprocess.DEVNULL)
            except OSError:
                self.stream.close(); self.stream = None
        return self

    def __exit__(self, *error):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        if self.stream:
            self.stream.close()


def play(binary, models, fights, directory, search, workers, progress=lambda done, pid: None, sample_devices=True,
         gpu_attended=0, gpu_wide_attended=0, encoding='legacy', gpu_wide_model=None):
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
               '--telemetry', str(directory / 'telemetry.jsonl'),
               '--sims', str(search.sims), '--gumbel-scale', str(search.gumbel_scale),
               '--rollout-mix', str(search.rollout_mix), '--particles', str(search.particles)]
    if encoding != 'legacy':
        command += ['--encoding', encoding]
    if gpu_attended:
        command += ['--gpu-attended', str(gpu_attended)]
    if gpu_wide_model is not None:
        command += ['--gpu-wide-model', str(gpu_wide_model),
                    '--gpu-wide-attended', str(gpu_wide_attended)]
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
