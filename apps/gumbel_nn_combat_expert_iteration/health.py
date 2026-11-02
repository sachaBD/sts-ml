"""Read-only run health sampling. Histories are bounded, in-memory, and phase-local.

Machine counters are host/device-wide, process counters cover the trainer and its
reported child only. GPU busy time is not peak-compute efficiency. Native counters
are interval deltas, never repeated as independent observations between reports.
"""
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import tomllib

from agents.combat.nn.metrics import last_json
from .dashboard import running

INTERVAL = 2
RETENTION = 1800

# The API describes metrics so the renderer doesn't need a second metric registry.
METRICS = {
    'games_s': ('Games / second', '/s', 'pipeline', 'Completions during the sampling interval; bursty with concurrent games.'),
    'decisions_s': ('Completed-game decisions / second', '/s', 'pipeline', 'Decisions credited when games finish; not live search activity.'),
    'nn_s': ('NN evaluations / second', '/s', 'pipeline', 'GPU leaves plus oversized CPU rows / native report interval; retries counted once.'),
    'gpu_s': ('GPU evaluations / second', '/s', 'pipeline', 'GPU leaves / native report interval; excludes oversized CPU rows.'),
    'cpu_oversized_s': ('Oversized CPU rows / second', '/s', 'pipeline', 'Rows exceeding GPU input capacity / native report interval.'),
    'precision_retry_s': ('Precision retries / second', '/s', 'pipeline', 'GPU predictions retried on CPU / native report interval.'),
    'cpu': ('Host CPU busy', '%', 'cpu', 'Across host logical CPUs; excludes idle and I/O wait. Not run-specific.'),
    'process_cpu': ('Run CPU consumption', 'cores', 'cpu', 'Trainer + reported child CPU seconds / wall second. One fully busy thread = 1 core.'),
    'iowait': ('Host CPU I/O wait', '%', 'cpu', 'Linux CPU accounting, not a measurement of this run’s blocking time.'),
    'gpu': ('GPU busy time', '%', 'gpu', 'Device 0 kernel-active time from NVML; not percent of peak FLOPs.'),
    'gpu_memory': ('GPU memory activity', '%', 'gpu', 'Device 0 memory-active time; not percent of peak bandwidth.'),
    'batch_fill': ('GPU batch fill', '%', 'gpu', 'Real GPU leaves / padded capacity in the native report interval.'),
    'fallback': ('CPU fallback requests', '%', 'gpu', 'Oversized rows plus GPU-to-CPU precision retries / all inference requests.'),
    'ram': ('Host RAM used', 'GiB', 'memory', 'MemTotal − MemAvailable; includes other processes.'),
    'rss': ('Run resident memory', 'GiB', 'memory', 'Sum of trainer + child RSS; shared pages may be counted twice.'),
    'vram': ('GPU memory used', 'GiB', 'memory', 'Device-wide allocation, not run-specific.'),
    'swap': ('Host swap used', 'GiB', 'memory', 'SwapTotal − SwapFree; occupancy alone is not current swapping activity.'),
    'read_mib': ('Run disk reads', 'MiB/s', 'io', 'Process read_bytes delta; cached reads are excluded.'),
    'write_mib': ('Run disk writes', 'MiB/s', 'io', 'Process write_bytes delta; includes kernel writeback accounting.'),
    'disk_free': ('Filesystem free', 'GiB', 'io', 'Available space on the run filesystem; no unmeasured runway extrapolation.'),
}


def distribution(values, current, bins=20):
    """Empirical reference range, not a confidence interval; current excluded by caller."""
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return dict(n=0, p10=None, median=None, p90=None, percentile=None, x=[], counts=[])

    def quantile(p):
        index = (len(values) - 1) * p
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)

    lo, hi = values[0], values[-1]
    if hi == lo:
        width = max(abs(lo) * .02, .01)
        lo, hi = lo - width, hi + width
    step = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, max(0, int((value - lo) / step)))] += 1
    return dict(n=len(values), p10=quantile(.1), median=quantile(.5), p90=quantile(.9),
                percentile=100 * sum(v <= current for v in values) / len(values) if current is not None else None,
                x=[lo + (i + .5) * step for i in range(bins)], counts=counts)


class Lines:
    """Incremental bounded reads; partial lines wait for their newline, rotations reset."""
    def __init__(self, limit=1024 * 1024):
        self.limit, self.offset, self.identity = limit, 0, None
        self.backlog = False

    def read(self, path):
        try:
            with Path(path).open('rb') as stream:
                stat = os.fstat(stream.fileno())
                identity = (str(path), stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.offset, self.identity = 0, identity
                stream.seek(self.offset)
                raw = stream.read(self.limit)
                self.backlog = stream.tell() < stat.st_size
                end = raw.rfind(b'\n') + 1
                if not end:
                    return []
                self.offset += end
                return raw[:end].decode(errors='replace').splitlines()
        except OSError:
            return []


def read_object(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def tail(path, limit=128 * 1024):
    try:
        with Path(path).open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - limit))
            raw = stream.read(limit).decode(errors='replace')
            return raw if size <= limit else raw.partition('\n')[2]
    except OSError:
        return ''


class Machine:
    def __init__(self, proc=Path('/proc')):
        self.proc, self.previous_cpu, self.previous_process = proc, {}, {}

    def sample(self, pids, now):
        values, details, cores = {}, {}, []
        try:
            for line in (self.proc / 'stat').read_text().splitlines():
                parts = line.split()
                if not parts or not parts[0].startswith('cpu'):
                    continue
                name, ticks = parts[0], list(map(int, parts[1:9]))
                total, idle, wait = sum(ticks), ticks[3], ticks[4]
                previous = self.previous_cpu.get(name)
                self.previous_cpu[name] = (total, idle, wait)
                busy = None
                if previous and total > previous[0]:
                    elapsed = total - previous[0]
                    busy = 100 * max(0, elapsed - (idle - previous[1]) - (wait - previous[2])) / elapsed
                    if name == 'cpu':
                        values.update(cpu=busy, iowait=100 * max(0, wait - previous[2]) / elapsed)
                if name != 'cpu':
                    cores.append(busy)
            details['cores'] = cores
            mem = {k: int(v.split()[0]) for k, v in
                   (line.split(':', 1) for line in (self.proc / 'meminfo').read_text().splitlines())}
            values.update(ram=(mem['MemTotal'] - mem['MemAvailable']) / 1024**2,
                          swap=(mem['SwapTotal'] - mem['SwapFree']) / 1024**2)
            details.update(ram_total=mem['MemTotal'] / 1024**2,
                           ram_available=mem['MemAvailable'] / 1024**2,
                           swap_total=mem['SwapTotal'] / 1024**2)
        except (OSError, ValueError, KeyError, IndexError):
            pass

        cpu, rss, reads, writes, cpu_n, io_n, processes = 0, 0, 0, 0, 0, 0, []
        next_process = {}
        for pid in sorted({int(p) for p in pids if p and int(p) > 0}):
            try:
                # comm can contain spaces and parentheses; fields after its final ')' are stable.
                stat = (self.proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
                if stat[0] == 'Z':
                    continue
                ticks, start = int(stat[11]) + int(stat[12]), int(stat[19])
                resident = int(stat[21]) * os.sysconf('SC_PAGE_SIZE') / 1024**3
                io = {}
                try:
                    io = dict(line.split(': ') for line in (self.proc / str(pid) / 'io').read_text().splitlines())
                except OSError:
                    pass
                read, write = (int(io[k]) if k in io else None for k in ('read_bytes', 'write_bytes'))
                key = (pid, start)
                previous = self.previous_process.get(key)
                next_process[key] = (now, ticks, read, write)
                if previous and now > previous[0]:
                    elapsed = now - previous[0]
                    cpu += max(0, ticks - previous[1]) / os.sysconf('SC_CLK_TCK') / elapsed
                    cpu_n += 1
                    if read is not None and previous[2] is not None:
                        reads += max(0, read - previous[2]) / elapsed / 1024**2
                        writes += max(0, write - previous[3]) / elapsed / 1024**2
                        io_n += 1
                rss += resident
                processes.append(dict(pid=pid, threads=int(stat[17])))
            except (OSError, ValueError, IndexError):
                continue
        self.previous_process = next_process
        if processes:
            values['rss'] = rss
        if cpu_n:
            values['process_cpu'] = cpu
        if io_n:
            values.update(read_mib=reads, write_mib=writes)
        details['processes'] = processes
        try:
            fields = 'utilization.gpu,utilization.memory,memory.used,memory.total,name'
            result = subprocess.run(['nvidia-smi', '--id=0', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=1.5, check=True)
            gpu, activity, used, total, name = result.stdout.strip().split(', ', 4)
            for key, raw in [('gpu', gpu), ('gpu_memory', activity), ('vram', used)]:
                try:
                    value = float(raw)
                    if math.isfinite(value):
                        values[key] = value / 1024 if key == 'vram' else value
                except ValueError:
                    pass
            details.update(vram_total=float(total) / 1024, gpu_name=name)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return values, details


def inference_delta(previous, current):
    """Native interval metrics; rejected after counter resets or zero elapsed time."""
    if not previous or current.get('seconds', 0) <= previous.get('seconds', 0):
        return {}
    keys = ('leaves', 'cpu_leaves', 'cpu_precision_retries', 'capacity', 'batches',
            'h2d_seconds', 'inference_seconds', 'd2h_seconds')
    delta = {k: current[k] - previous[k] for k in keys if k in current and k in previous}
    if any(v < 0 for v in delta.values()):
        return {}
    elapsed = current['seconds'] - previous['seconds']
    values = {}
    if 'leaves' in delta and 'cpu_leaves' in delta:
        requests = delta['leaves'] + delta['cpu_leaves']
        values['nn_s'] = requests / elapsed
        values['gpu_s'] = delta['leaves'] / elapsed
        values['cpu_oversized_s'] = delta['cpu_leaves'] / elapsed
        if 'cpu_precision_retries' in delta:
            values['precision_retry_s'] = delta['cpu_precision_retries'] / elapsed
        if requests and 'cpu_precision_retries' in delta:
            values['fallback'] = 100 * (delta['cpu_leaves'] + delta['cpu_precision_retries']) / requests
    if delta.get('capacity'):
        values['batch_fill'] = 100 * delta['leaves'] / delta['capacity']
    return values


class Health:
    def __init__(self, out):
        self.out = Path(out)
        self.samples = deque(maxlen=RETENTION // INTERVAL + 1)
        self.events = deque(maxlen=200)
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.machine = Machine()
        self.result_lines, self.event_lines = Lines(), Lines()
        self.worker_lines = Lines(limit=128 * 1024)
        self.stage_key, self.previous_native, self.previous_sample = None, {}, None
        self.native_interval = None
        self.games, self.actions, self.result_decks = 0, 0, set()
        self.durations = deque(maxlen=10000)
        self.action_counts = deque(maxlen=10000)
        self.config, self.config_hash, self.config_stamp = {}, '', None
        self.decks, self.started, self.latest = None, time.time(), None
        self.last_completion = None

    def event(self, level, text, source, now):
        with self.lock:
            self.events.append(dict(time=now, level=level, text=text[:2000], source=source))

    def load_config(self):
        path = self.out / 'config.toml'
        stamp = path.stat().st_mtime_ns
        if stamp != self.config_stamp:
            raw = path.read_bytes()
            self.config = tomllib.loads(raw.decode())
            self.config_hash, self.config_stamp = hashlib.sha256(raw).hexdigest()[:12], stamp
        if self.decks is None:
            path = self.out / 'inputs/starts.json'
            if path.exists():
                # Frozen input; decode once, discard states rather than retain 10k decks in the dashboard.
                self.decks = len(json.loads(path.read_text()))

    def sample(self):
        now = time.time()
        self.load_config()
        progress = read_object(self.out / 'training/status.json')
        active = running(self.out / 'training')
        phase = progress.get('phase', 'preparing')
        stage = progress.get('stage', '')
        state = ('stopping' if (self.out / 'training/STOP').exists() else 'running') if active else (
            phase if phase in ('done', 'stopped') else 'interrupted' if progress else 'preparing')
        cohort = ('evaluation' if stage.startswith('test') else 'self-play') if phase == 'play' else phase
        if not active:
            cohort = state
        stage_key = (stage, phase, progress.get('directory'), progress.get('child_pid'), progress.get('pid'), active)
        if stage_key != self.stage_key:
            self.event('info', f'{state} · {stage or "run"} / {phase}', 'status', now)
            self.stage_key = stage_key
            self.previous_native, self.previous_sample = {}, None
            self.native_interval = None
            self.games, self.actions, self.result_decks = 0, 0, set()
            self.last_completion = None
            self.durations.clear()
            self.action_counts.clear()
            self.result_lines, self.worker_lines = Lines(), Lines(limit=128 * 1024)
        pids = [progress.get('pid'), progress.get('child_pid')] if active else []
        values, machine = self.machine.sample(pids, now)
        disk = shutil.disk_usage(self.out)
        values['disk_free'] = disk.free / 1024**3
        machine.update(disk_total=disk.total / 1024**3, disk_used=disk.used / 1024**3)
        directory = Path(progress['directory']) if progress.get('directory') else None
        native = {}
        if directory and phase == 'play':
            for line in self.result_lines.read(directory / 'results.jsonl'):
                try:
                    row = json.loads(line)
                    self.games += 1
                    self.actions += row.get('decisions', 0)
                    self.result_decks.add(row['fight_id'].split(':')[-2])
                    if isinstance(row.get('seconds'), (int, float)):
                        self.durations.append(row['seconds'])
                    self.action_counts.append(row.get('decisions', 0))
                    self.last_completion = max(self.last_completion or 0, row.get('finished_unix', 0)) or None
                except (ValueError, KeyError, TypeError):
                    continue
            if active and self.previous_sample:
                then, games, actions, catching_up = self.previous_sample
                if now > then and not catching_up and not self.result_lines.backlog:
                    values.update(games_s=(self.games - games) / (now - then),
                                  decisions_s=(self.actions - actions) / (now - then))
            self.previous_sample = (now, self.games, self.actions, self.result_lines.backlog)
            text = tail(directory / 'worker.log')
            native = {prefix: last_json(text, prefix) for prefix in ('gpu_stats', 'actors_stats', 'play_stats')}
            current = native['gpu_stats']
            if active and current != self.previous_native:
                interval_values = inference_delta(self.previous_native, current)
                values.update(interval_values)
                self.native_interval = (dict(seconds=current['seconds'] - self.previous_native['seconds'],
                                             requests=sum(current[k] - self.previous_native[k]
                                                          for k in ('leaves', 'cpu_leaves'))
                                             if 'nn_s' in interval_values else None,
                                             observed=now) if interval_values else None)
                self.previous_native = current
            for line in self.worker_lines.read(directory / 'worker.log'):
                if any(word in line.lower() for word in ('error', 'failed', 'retry', 'retried', 'warning', 'non-finite')):
                    self.event('warning', line, 'worker · observed', now)
        for line in self.event_lines.read(self.out / 'training/log.jsonl'):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # Native run log rows have no wall-clock timestamp. Do not invent one.
            self.event('info', json.dumps(row, ensure_ascii=False), 'run log · observed', now)
        recovery = read_object(self.out / 'recovery.json')
        values = {k: v for k, v in values.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        latest = dict(time=now, state=state, progress=progress, cohort=cohort, stage_id=repr(stage_key),
                      machine=machine, native=native, native_interval=self.native_interval, recovery=recovery,
                      games=self.games, actions=self.actions, decks_seen=len(self.result_decks),
                      last_completion=self.last_completion, catching_up=self.result_lines.backlog,
                      config=self.config, config_hash=self.config_hash, decks=self.decks,
                      duration=distribution(list(self.durations), None),
                      actions_per_game=distribution(list(self.action_counts), None))
        with self.lock:
            self.samples.append(dict(time=now, cohort=cohort, config=self.config_hash,
                                     stage_id=repr(stage_key), values=values))
            self.latest = latest

    def current(self):
        """Cached measurements only; no history aggregation or subprocesses on HTTP threads."""
        with self.lock:
            if self.latest is None:
                return None
            return dict(self.latest, values=dict(self.samples[-1]['values']))

    def view(self, window=900):
        now = time.time()
        with self.lock:
            samples, latest = list(self.samples), self.latest
            events = list(self.events)
        if latest is None:
            return dict(version=1, ready=False, events=events)
        window = min(RETENTION, max(60, window))
        comparable = [s for s in samples if s['time'] >= now - window and
                      s['cohort'] == latest['cohort'] and s['config'] == latest['config_hash']]
        metrics = {}
        for key, (label, unit, group, description) in METRICS.items():
            points = [(s['time'], s['values'][key]) for s in comparable if key in s['values']]
            current_points = [(s['time'], s['values'][key]) for s in comparable
                              if s['stage_id'] == latest['stage_id'] and key in s['values']]
            stamp, current = current_points[-1] if current_points else (None, None)
            age = now - stamp if stamp is not None else None
            stale = age is not None and age > (130 if key in
                ('nn_s', 'gpu_s', 'cpu_oversized_s', 'precision_retry_s', 'batch_fill', 'fallback') else 10)
            metrics[key] = dict(label=label, unit=unit, group=group, description=description,
                                current=current, age=age, stale=stale,
                                reference=distribution([v for t, v in points if t != stamp], None if stale else current),
                                times=[t for t, _ in points], values=[v for _, v in points])
        return dict(version=1, ready=True, **latest, metrics=metrics, events=events,
                    started=self.started, interval=INTERVAL, window=window,
                    config_path=str(self.out / 'config.toml'),
                    unavailable=['Live inference queue depth / wait latency', 'In-flight game ages',
                                 'Peak compute / memory-bandwidth efficiency', 'Live actor wait classification'])

    def run(self):
        while not self.stop.is_set():
            begin = time.monotonic()
            try:
                self.sample()
            except Exception as error:
                self.event('error', str(error), 'sampler', time.time())
            self.stop.wait(max(.1, INTERVAL - (time.monotonic() - begin)))
