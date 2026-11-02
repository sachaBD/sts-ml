"""Public dashboard API v1. Both live and mock providers use this contract.

Fractions are 0..1, storage is bytes, rates are /second, latency is milliseconds,
timestamps are Unix seconds. Null means unmeasured, never idle or zero usage.
"""
import time

from .execution import RuntimeCollector, finite

GIB = 1024**3


def envelope(source, run, status, *, captured_at=None, worker_instance_id=None, sequence=None,
             window=None, reason=None, configuration=None, data=None, generated_at=None):
    return dict(schema_version=1, source=source, status=status,
                generated_at=time.time() if generated_at is None else generated_at,
                captured_at=captured_at, worker_instance_id=worker_instance_id, sequence=sequence,
                window=window, reason=reason, run=run, configuration=configuration, data=data)


def run_context(config, progress, state):
    return dict(id=config.get('run', {}).get('id'), state=state, stage=progress.get('stage'),
                phase=progress.get('phase'), done=progress.get('done'), total=progress.get('total'))


def configuration(config):
    workers = config.get('workers', {})
    return dict(actor_count=workers.get('actors'), slots_per_actor=workers.get('slots'),
                batch_cap=workers.get('gpu_batch'), gather_wait_microseconds=workers.get('gpu_wait_us'))


def resource_data(values, machine, path):
    def fraction(key):
        value = values.get(key)
        return min(1, value / 100) if finite(value) else None

    def size(value):
        return round(value * GIB) if finite(value) else None

    return dict(
        cpu=dict(scope='host', utilisation_fraction=fraction('cpu'),
                 logical_cores=[dict(id=i, utilisation_fraction=min(1, value / 100) if finite(value) else None)
                                for i, value in enumerate(machine.get('cores', []))]),
        gpu=dict(scope='device', device_index=0, name=machine.get('gpu_name'),
                 utilisation_fraction=fraction('gpu'),
                 memory_used_bytes=size(values.get('vram')), memory_total_bytes=size(machine.get('vram_total'))),
        memory=dict(scope='host', used_bytes=size(values.get('ram')),
                    available_bytes=size(machine.get('ram_available')), total_bytes=size(machine.get('ram_total'))),
        disk=dict(path=path, used_bytes=size(machine.get('disk_used')),
                  available_bytes=size(values.get('disk_free')), total_bytes=size(machine.get('disk_total'))))


def mock_execution(legacy):
    topology = legacy.get('topology')
    if topology is None:
        return None
    return dict(
        actors=[dict(id=a['id'], capacity_slots=224,
                     slots=dict(ready_mean=a['ready'], awaiting_inference_mean=a['waiting'], empty_mean=a['empty']),
                     working_fraction=a['working_fraction'], requests_per_second=a['requests_s'],
                     completed_games_per_second=None) for a in topology['actors']],
        queues=[dict(id=0, backend_id=0, capacity_requests=topology['queue']['capacity'],
                     mean_depth=topology['queue']['depth'], peak_depth=topology['queue']['peak'],
                     current_depth=None, oldest_request_ms=None,
                     wait=dict(method='synthetic_quantiles', samples=topology['queue']['samples'],
                               **{key: dict(lower_ms=topology['queue'][field], upper_ms=topology['queue'][field])
                                  for key, field in [('p50', 'wait_p50_ms'), ('p95', 'wait_p95_ms')]}))],
        dispatchers=[dict(id=d['id'], backend_id=0, requests_per_second=d['requests_s'],
                          batches_per_second=d['batches_s'], batch_fill_fraction=d['batch_fill'])
                     for d in topology['dispatchers']],
        backends=[dict(id=0, kind='gpu', max_batch_rows=512, token_capacity=64, attention_capacity=24,
                       lanes=[dict(id=lane['id'], occupied_fraction=lane['occupied_fraction'],
                                   time_fractions=dict(lane['time_fractions'], cpu_retry=0),
                                   evaluations_per_second=lane['evals_s'], batches_per_second=lane['batches_s'],
                                   batch_fill_fraction=lane['batch_fill']) for lane in topology['lanes']])],
        fallback=dict(share_fraction=legacy['metrics'].get('fallback', {}).get('current', 0) / 100,
                      oversized_per_second=None, precision_retries_per_second=None),
        demo_mode=topology['mode'])


class DashboardAPI:
    def __init__(self, health, out=None):
        self.health, self.out = health, out
        self.source = 'native' if out else 'mock'
        self.runtime = RuntimeCollector(out) if out else None

    def _snapshot(self):
        return self.health.current() if self.runtime else self.health.view()

    def resources(self):
        now, snapshot = time.time(), self._snapshot()
        if snapshot is None or snapshot.get('ready') is False:
            return envelope(self.source, None, 'warming', reason='waiting_for_machine_sample', generated_at=now)
        run = run_context(snapshot['config'], snapshot.get('progress', {}), snapshot['state'])
        captured = snapshot['time']
        stale = now - captured > 6
        if self.runtime:
            values = snapshot['values']
        else:
            values = {key: metric['current'] for key, metric in snapshot['metrics'].items() if not metric.get('stale')}
        return envelope(self.source, run, 'stale' if stale else 'live', captured_at=captured,
                        reason='machine_sample_stale' if stale else None, generated_at=now,
                        data=None if stale else resource_data(values, snapshot['machine'], str(self.out) if self.out else 'mock'))

    def execution(self):
        now, snapshot = time.time(), self._snapshot()
        if snapshot is None or snapshot.get('ready') is False:
            return envelope(self.source, None, 'warming', reason='waiting_for_run_context', generated_at=now)
        config, progress = snapshot['config'], snapshot.get('progress', {})
        run = run_context(config, progress, snapshot['state'])
        capacity = configuration(config)
        if self.runtime:
            result = self.runtime.view(progress, config.get('workers', {}), now=now)
        else:
            topology = snapshot.get('topology')
            stale = topology is not None and now - topology['time'] > 5
            result = dict(status='stale' if stale else 'live' if topology else 'unavailable',
                          reason='worker_snapshot_stale' if stale else None if topology else 'worker_telemetry_unavailable',
                          captured_at=topology['time'] if topology else None, worker_instance_id='mock-worker',
                          sequence=int(now - self.health.started),
                          window=dict(requested_seconds=5, observed_seconds=5 if topology else 0,
                                      samples=26 if topology else 0, missing_samples=0),
                          data=mock_execution(snapshot))
        if snapshot['state'] not in ('running', 'stopping') or progress.get('phase') != 'play':
            result.update(status='idle', reason='not_collecting', data=None)
        if now - snapshot['time'] > 6:
            result.update(status='stale', reason='run_context_stale', data=None)
        if result['status'] in ('stale', 'unavailable'):
            result['data'] = None
        if result['data'] is not None:
            result['data'].pop('configuration', None)
            result['data'].setdefault('demo_mode', None)
        return envelope(self.source, run, configuration=capacity, generated_at=now, **result)
