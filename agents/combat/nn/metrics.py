"""Standard performance metrics. Missing counters stay None, never inferred from unrelated utilization.

Decisions = searched game decisions, NOT NN leaves. Worker CPU is normalized by all allocated
SimWorkers and whole gameplay wall time (including the tail). Coordinator non-CPU time includes
waiting/descheduling; it is not proven I/O time. NVML GPU busy is active TIME, not peak compute.
VRAM occupancy, memory-controller active time, bandwidth efficiency and SM throughput are distinct.
"""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess


METRIC_DEFINITIONS = {
    'decisions_per_second': 'searched game decisions / gameplay seconds',
    'nn_evaluations_per_second': 'real evaluated leaves (GPU + CPU fallback) / gameplay seconds; excludes padding',
    'games_per_hour_gameplay': 'completed games / gameplay seconds * 3600',
    'games_per_hour_full_update': 'completed games / full collect-load-train-save-export seconds * 3600',
    'worker_cpu_pct': 'sum SimWorker thread CPU seconds / (SimWorkers * gameplay wall seconds) * 100',
    'coordinator_cpu_pct': 'coordinator thread CPU seconds / gameplay wall seconds * 100',
    'coordinator_non_cpu_pct': '100 - coordinator CPU%; includes waits/descheduling, not strict I/O-bound time',
    'coordinator_gpu_api_pct': 'wall time inside H2D/inference/D2H calls / gameplay seconds; overlaps CPU time when driver busy-spins',
    'coordinator_io_bound_pct': 'requires wait classification/tracing; not inferred from CPU time or GPU API time',
    'gpu_busy_pct': 'NVML percent time a kernel was active; NOT percent of peak FLOPs',
    'gpu_vram_pct': 'device-wide used VRAM / device VRAM capacity * 100',
    'gpu_memory_activity_pct': 'NVML percent time global memory was read/written; NOT peak bandwidth fraction',
    'gpu_memory_bandwidth_pct': 'requires hardware profiling: measured bandwidth relative to peak; otherwise unavailable',
    'gpu_compute_throughput_pct': 'requires hardware profiling: SM/compute throughput relative to peak; otherwise unavailable',
}


class DeviceSampler:
    """Optional per-stage NVML CSV. Missing nvidia-smi leaves GPU counters unavailable."""
    def __init__(self, path, enabled=True):
        self.path, self.enabled = Path(path), enabled
        self.process = self.stream = None

    def __enter__(self):
        binary = shutil.which('nvidia-smi') if self.enabled else None
        if binary:
            self.stream = self.path.open('w')
            fields = 'timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw'
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


def last_json(text, prefix):
    for line in reversed(text.splitlines()):
        if line.startswith(prefix + ': '):
            try:
                return json.loads(line[len(prefix)+2:])
            except json.JSONDecodeError:
                continue # live log may end halfway through a write
    return {}


def play_metrics(directory):
    directory = Path(directory)
    path = directory/'worker.log'
    text = path.read_text(errors='replace') if path.exists() else ''
    play, runtime, gpu = (last_json(text,p) for p in ('play_stats','runtime_stats','gpu_stats'))
    seconds = play.get('seconds')
    if not seconds:
        match = re.search(r'timing: gameplay ([\d.]+)',text)
        seconds = float(match[1]) if match else None
    rows = []
    if (directory/'results.jsonl').exists():
        for line in (directory/'results.jsonl').read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    decisions = play.get('searched_decisions',sum(r['decisions'] for r in rows))
    games = play.get('games',len(rows))
    result = {k:None for k in METRIC_DEFINITIONS}
    nn_count = gpu['leaves'] + gpu.get('cpu_leaves',0) if 'leaves' in gpu else None
    if nn_count is None:
        match = re.search(r'coordinator: \d+ batches, (\d+) leaves',text)
        nn_count = int(match[1]) if match else None
    if seconds:
        result.update(decisions_per_second=decisions/seconds,
                      games_per_hour_gameplay=games*3600/seconds,
                      nn_evaluations_per_second=nn_count/seconds if nn_count is not None else None)
    for key in ('worker_cpu_pct','coordinator_cpu_pct','coordinator_non_cpu_pct'):
        result[key] = runtime.get(key)
    if seconds and all(k in gpu for k in ('h2d_seconds','inference_seconds','d2h_seconds')):
        result['coordinator_gpu_api_pct'] = 100*sum(gpu[k] for k in ('h2d_seconds','inference_seconds','d2h_seconds'))/seconds
    result['batch_fill_pct'] = 100*gpu['leaves']/gpu['capacity'] if gpu.get('capacity') else None
    result['gameplay_wall_start'] = play.get('wall_start_unix')
    result['gameplay_seconds'] = seconds
    return result


def device_metrics(telemetry, capacity_mib=None):
    """Map a phase's sampled NVML means; no NVML counter is mislabeled as peak efficiency."""
    get = lambda name: telemetry.get(name,{}).get('mean')
    vram = get('vram_mib')
    return dict(gpu_busy_pct=get('gpu_util'),
                gpu_vram_pct=100*vram/capacity_mib if vram is not None and capacity_mib else None,
                gpu_memory_activity_pct=get('memory_util'),
                gpu_memory_bandwidth_pct=None, gpu_compute_throughput_pct=None)


def sampled_devices(path, start, end, capacity_mib=None):
    """Low-overhead NVML means for an explicit wall-clock interval; safe on a live CSV."""
    path = Path(path)
    rows = []
    if path.exists() and start is not None and end is not None:
        with path.open() as stream:
            for raw in csv.DictReader(stream):
                try:
                    row = {k.strip():v.strip() for k,v in raw.items()}
                    stamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                    if start <= stamp <= end:
                        rows.append(row)
                except (ValueError, TypeError, AttributeError, KeyError):
                    continue
    if capacity_mib is None:
        for row in rows:
            try: capacity_mib = float(row['memory.total [MiB]']); break
            except (KeyError,ValueError): continue
    telemetry = {}
    for label, field in dict(gpu_util='utilization.gpu [%]', memory_util='utilization.memory [%]',
                             vram_mib='memory.used [MiB]',power_w='power.draw [W]').items():
        values = []
        for row in rows:
            try: values.append(float(row[field]))
            except (KeyError,ValueError): continue
        if values:
            telemetry[label]=dict(mean=sum(values)/len(values),minimum=min(values),maximum=max(values))
    return dict(samples=len(rows), metrics=device_metrics(telemetry,capacity_mib),telemetry=telemetry)


def performance_header():
    return 'iter   dec/s   NN eval/s   games/h*  workerCPU coordCPU/GPUapi  GPUbusy  VRAM  memBW  SMpeak'


def performance_row(iteration, metrics):
    def n(key):
        v=metrics.get(key)
        return f'{v:,.0f}' if v is not None else '—'
    def pct(key):
        v=metrics.get(key)
        return f'{v:.0f}%' if v is not None else '—'
    coord=pct('coordinator_cpu_pct')+'/'+pct('coordinator_gpu_api_pct')
    return (f'{iteration:>4} {n("decisions_per_second"):>7} {n("nn_evaluations_per_second"):>11} '
            f'{n("games_per_hour_full_update"):>10} {pct("worker_cpu_pct"):>10} {coord:>14} '
            f'{pct("gpu_busy_pct"):>8} {pct("gpu_vram_pct"):>5} '
            f'{pct("gpu_memory_bandwidth_pct"):>6} {pct("gpu_compute_throughput_pct"):>7}')
