"""A training run's life on disk: one process at a time, live status, an append-only log, a clean stop,
a disk reserve and safe resumption. Knows iteration directories, not what happens inside them.

  DIR/.lock          held while a process owns the run (the dashboard tests it for liveness)
  DIR/status.json    the current stage and phase, rewritten atomically
  DIR/log.jsonl      one row per completed iteration, evaluation or event
  DIR/STOP           requests a clean stop at the next iteration boundary
  DIR/iterNNN/       an iteration's samples, results and checkpoint; complete once logged
"""
import contextlib
import fcntl
import json
import os
import shutil
import time


class RunControl:
    def __init__(self, directory):
        self.directory = directory

    def iteration(self, k):
        return self.directory / f'iter{k:03d}'

    @contextlib.contextmanager
    def lock(self):
        with (self.directory / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if another process owns the run
            yield

    def status(self, **fields):
        partial = self.directory / '.status.json'
        partial.write_text(json.dumps(dict(time=time.time(), pid=os.getpid(), **fields)))
        partial.replace(self.directory / 'status.json')

    def log(self, **row):
        with (self.directory / 'log.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def history(self):
        path = self.directory / 'log.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def stop_requested(self):
        return (self.directory / 'STOP').exists()

    def check_resume(self, completed):
        """Resuming must continue from the last completed iteration, and never append to a partial one."""
        logged = [r['iteration'] for r in self.history() if 'iteration' in r]
        if not logged or completed != max(logged):
            raise ValueError(f'resume from the latest completed iteration ({max(logged, default=None)}), not {completed}')
        if self.iteration(completed + 1).exists():
            raise FileExistsError('archive the interrupted iteration before resuming; never append partial samples')

    def archive(self, directory):
        """Moves a partial stage aside (kept for diagnosis) so it can be redone from scratch."""
        if directory.exists():
            directory.rename(directory.with_name(f'.{directory.name}-interrupted-{time.time_ns()}'))

    def disk_reserve(self, min_free_gib, previous):
        """Bytes needed before the next iteration: the reserve plus room for one more iteration like `previous`."""
        samples = sum(p.stat().st_size for p in previous.glob('samples-*.bin'))
        needed = min_free_gib * 1024**3 + max(3 * 1024**3, samples)
        return needed if shutil.disk_usage(self.directory).free < needed else None

    def prune(self, keep_from):
        """Deletes samples and TensorRT caches of iterations before `keep_from`; checkpoints and results stay."""
        for old in self.directory.glob('iter[0-9]*'):
            if old.is_dir() and old.name[4:].isdigit() and int(old.name[4:]) < keep_from:
                for samples in old.glob('samples-*.bin'):
                    samples.unlink()
                    samples.with_suffix('.index.npz').unlink(missing_ok=True)
                shutil.rmtree(old / 'tensorrt', ignore_errors=True)

    @staticmethod
    def delete_samples(directory):
        for samples in directory.rglob('samples-*.bin'):
            samples.unlink()
