"""Plays fights with the native nn_play: one process per call, so each stage gets a fresh CUDA context
(a crash is recoverable by a new process) and never shares GPU state with the PyTorch learner.

nn_play's settings are a table of [search] and [workers] tables; nn_play alone knows their keys (play_config.hpp)."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import tomli_w


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


def check(binary, settings):
    """Raises ValueError with nn_play's reason unless `settings` would play."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'play.toml'
        path.write_text(tomli_w.dumps(settings))
        checked = subprocess.run([str(binary), str(path), '--check'], capture_output=True, text=True)
    if checked.returncode:
        raise ValueError(checked.stderr.strip())


def play(binary, models, fights, directory, settings, *, progress=lambda done, pid: None, sample_devices=True):
    """Plays `fights` with nn_play `binary`, the exported model directory `models` and `settings`; writes the
    settings (play.toml), samples, results and logs to `directory`. Results are journaled as fights finish;
    progress(done, pid) follows them."""
    directory.mkdir(parents=True, exist_ok=True)
    samples = directory / 'samples-0.bin'
    if samples.exists():
        raise FileExistsError(f'{samples}: refusing to append duplicate gameplay')
    starts = directory / 'starts.jsonl'
    starts.write_text(''.join(json.dumps(f) + '\n' for f in fights))
    (directory / 'play.toml').write_text(tomli_w.dumps(settings))
    command = [str(binary), str(directory / 'play.toml'), str(models), str(samples),
               '--telemetry', str(directory / 'telemetry.jsonl')]
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
