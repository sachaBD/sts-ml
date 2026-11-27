"""Plays fights with the native nn_play: one process per call, so each stage gets a fresh CUDA context
(a crash is recoverable by a new process) and never shares GPU state with the PyTorch learner.

nn_play's settings pass through as written; agents/combat/player/play_config.hpp owns them."""
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


def gpu_environment():
    """os.environ with the dependency-managed TensorRT and CUDA libraries, not system-wide installs, on the loader path."""
    packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
    return dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))


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
    env = gpu_environment()
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


def publish_partition(directory, fights, results, samples, *, retain_samples):
    """Publish an ordinary per-set results journal with per-set byte offsets."""
    directory.mkdir(parents=True)
    (directory / 'starts.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in fights))
    # nn_play writes one sample file; read spans without loading the batch's replay into RAM.
    offset = 0
    if retain_samples:
        with samples.open('rb') as source, (directory / samples.name).open('wb') as dest:
            for row in results:
                span = row['sample_range']
                source.seek(span['begin'])
                remaining = span['end'] - span['begin']
                while remaining > 0:
                    data = source.read(min(remaining, 1024 * 1024))
                    if not data:
                        raise ValueError(f'{samples}: truncated evaluation samples')
                    dest.write(data)
                    remaining -= len(data)
    for row in results:
        if span := row.get('sample_range'):
            end = offset + span['end'] - span['begin']
            row['sample_range'] = {**span, 'begin': offset, 'end': end}
            offset = end
    (directory / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
