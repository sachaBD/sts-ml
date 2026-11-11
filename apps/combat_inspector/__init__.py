"""Standalone combat decision inspection, using a persistent CPU inference process."""
import json
from pathlib import Path
import subprocess
import threading


class Inspector:
    """Use as a context manager. Calls are serialized; native errors raise ValueError."""

    def __init__(self, executable=None):
        self.executable = Path(executable) if executable else (
            Path(__file__).resolve().parents[2] / 'build/main/apps/combat_inspector/combat_explain')
        self._process = None
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def close(self):
        with self._lock:
            if self._process is not None:
                self._process.stdin.close()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                self._process.stdout.close()
                self._process = None

    def decision(self, *, model, encoding, start, moves=(), sims=64, particles=8,
                 gumbel_scale=0, rollout_mix=0):
        request = json.dumps(dict(model=str(Path(model).resolve()), encoding=encoding, start=start,
                                  moves=list(moves), search=dict(sims=sims, particles=particles,
                                  gumbel_scale=gumbel_scale, rollout_mix=rollout_mix)), allow_nan=False)
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                if self._process is not None:
                    self._process.stdin.close()
                    self._process.stdout.close()
                self._process = subprocess.Popen([str(self.executable)], stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, text=True)
            self._process.stdin.write(request + '\n')
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError('combat_explain exited without a response')
            result = json.loads(line)
        if 'error' in result:
            raise ValueError(result['error'])
        return result
