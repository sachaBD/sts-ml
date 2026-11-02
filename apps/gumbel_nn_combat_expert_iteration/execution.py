"""Native runtime counters -> bounded windows. No machine sampling or HTTP dependencies."""
from collections import deque
import copy
import json
import math
from pathlib import Path
import threading
import time

from .health import read_object

SLOT_STATES = ('ready', 'awaiting_inference', 'empty')
LANE_STATES = ('idle', 'packing', 'h2d', 'inference', 'd2h', 'cpu_retry')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_snapshot(raw):
    """Reject malformed native input before it can contaminate a counter window."""
    def require(condition):
        if not condition:
            raise ValueError('invalid runtime snapshot')

    def counter(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def rows(key, parent=raw):
        values = parent[key]
        require(isinstance(values, list) and len(values) <= 4096)
        require(all(isinstance(v, dict) and counter(v.get('id')) for v in values))
        require(len({v['id'] for v in values}) == len(values))
        return values

    try:
        require(raw['schema_version'] == 1 and isinstance(raw['worker_instance_id'], str) and bool(raw['worker_instance_id']))
        require(counter(raw['pid']) and raw['pid'] > 0 and counter(raw['sequence']))
        require(finite(raw['elapsed_seconds']) and finite(raw['captured_at']))
        for actor in rows('actors'):
            require(counter(actor['slots']) and actor['slots'] > 0)
            require(all(counter(actor[k]) for k in ('requests', 'completed')))
            require(finite(actor['working_seconds']))
            require(set(actor['slot_seconds']) == set(SLOT_STATES))
            require(all(finite(v) for v in actor['slot_seconds'].values()))
        backend_ids = set()
        for backend in rows('backends'):
            backend_ids.add(backend['id'])
            require(backend['kind'] in ('cpu', 'gpu', 'unknown'))
            require(all(counter(backend[k]) for k in ('max_batch', 'tokens', 'attended', 'oversized_rows', 'precision_retries')))
            for lane in rows('lanes', backend):
                require(all(counter(lane[k]) for k in ('rows', 'batches', 'capacity_rows')))
                require(set(lane['time_seconds']) == set(LANE_STATES))
                require(all(finite(v) for v in lane['time_seconds'].values()))
        for queue in rows('queues'):
            require(queue['id'] in backend_ids)
            require(counter(queue['depth']) and counter(queue['peak_depth_interval']))
            require(queue['peak_depth_interval'] >= queue['depth'])
            require(finite(queue['depth_seconds']) and finite(queue['oldest_seconds']))
            histogram = queue['wait_histogram']
            bounds, counts = histogram['upper_bounds_seconds'], histogram['counts']
            require(isinstance(bounds, list) and 0 < len(bounds) <= 128)
            require(all(finite(v) and v > 0 for v in bounds))
            require(all(a < b for a, b in zip(bounds, bounds[1:])))
            require(isinstance(counts, list) and len(counts) == len(bounds) + 1 and all(counter(v) for v in counts))
        for dispatcher in rows('dispatchers'):
            require(dispatcher['backend_id'] in backend_ids)
            require(all(counter(dispatcher[k]) for k in ('requests', 'batches', 'capacity_rows')))
    except (KeyError, TypeError, AttributeError, OverflowError):
        raise ValueError('invalid runtime snapshot') from None
    return raw


def keyed(rows):
    return {row['id']: row for row in rows}


def difference(current, previous):
    value = current - previous
    if value < -1e-8:
        raise ValueError('counter reset')
    return max(0, value)


def quantile_bounds(bounds, counts, fraction):
    count = sum(counts)
    if not count:
        return None
    rank, cumulative = max(1, math.ceil(fraction * count)), 0
    for i, n in enumerate(counts):
        cumulative += n
        if cumulative >= rank:
            return dict(lower_ms=1000 * bounds[i - 1] if i else 0,
                        upper_ms=1000 * bounds[i] if i < len(bounds) else None)
    return None


def aggregate(first, last, snapshots, config):
    elapsed = last['elapsed_seconds'] - first['elapsed_seconds']
    if elapsed <= 0:
        raise ValueError('non-increasing runtime clock')
    gaps = last['sequence'] - first['sequence'] + 1 - len(snapshots)
    result = dict(configuration=dict(actor_count=len(last['actors']),
                  slots_per_actor=config.get('slots'), batch_cap=config.get('gpu_batch'),
                  gather_wait_microseconds=config.get('gpu_wait_us')),
                  actors=[], queues=[], dispatchers=[], backends=[], fallback=None)
    for collection in ('actors', 'queues', 'dispatchers', 'backends'):
        if set(keyed(first[collection])) != set(keyed(last[collection])):
            raise ValueError('topology changed within worker instance')
    old_actors = keyed(first['actors'])
    for actor in last['actors']:
        old = old_actors[actor['id']]
        if old['slots'] != actor['slots']:
            raise ValueError('topology changed within worker instance')
        result['actors'].append(dict(id=actor['id'], capacity_slots=actor['slots'],
            slots={f'{key}_mean': difference(actor['slot_seconds'][key], old['slot_seconds'][key]) / elapsed for key in SLOT_STATES},
            working_fraction=min(1, difference(actor['working_seconds'], old['working_seconds']) / elapsed),
            requests_per_second=difference(actor['requests'], old['requests']) / elapsed,
            completed_games_per_second=difference(actor['completed'], old['completed']) / elapsed))
    old_queues = keyed(first['queues'])
    for queue in last['queues']:
        old = old_queues[queue['id']]
        histogram, previous_histogram = queue['wait_histogram'], old['wait_histogram']
        bounds = histogram['upper_bounds_seconds']
        if bounds != previous_histogram['upper_bounds_seconds']:
            raise ValueError('histogram changed within worker instance')
        counts = [int(difference(a, b)) for a, b in zip(histogram['counts'], previous_histogram['counts'])]
        result['queues'].append(dict(id=queue['id'], backend_id=queue['id'],
            capacity_requests=sum(a['slots'] for a in last['actors']),
            mean_depth=difference(queue['depth_seconds'], old['depth_seconds']) / elapsed,
            peak_depth=None if gaps else max(keyed(s['queues'])[queue['id']]['peak_depth_interval'] for s in snapshots[1:]),
            current_depth=queue['depth'], oldest_request_ms=queue['oldest_seconds'] * 1000,
            wait=dict(method='histogram_bounds', samples=sum(counts),
                      p50=quantile_bounds(bounds, counts, .5), p95=quantile_bounds(bounds, counts, .95))))
    old_dispatchers = keyed(first['dispatchers'])
    total_requests = 0
    for dispatcher in last['dispatchers']:
        old = old_dispatchers[dispatcher['id']]
        requests = difference(dispatcher['requests'], old['requests'])
        batches = difference(dispatcher['batches'], old['batches'])
        capacity = difference(dispatcher['capacity_rows'], old['capacity_rows'])
        total_requests += requests
        result['dispatchers'].append(dict(id=dispatcher['id'], backend_id=dispatcher['backend_id'],
            requests_per_second=requests / elapsed, batches_per_second=batches / elapsed,
            batch_fill_fraction=requests / capacity if capacity else None))
    old_backends, oversized, retries = keyed(first['backends']), 0, 0
    for backend in last['backends']:
        old = old_backends[backend['id']]
        oversized += difference(backend['oversized_rows'], old['oversized_rows'])
        retries += difference(backend['precision_retries'], old['precision_retries'])
        converted = dict(id=backend['id'], kind=backend['kind'], max_batch_rows=backend['max_batch'],
                         token_capacity=backend['tokens'] or None, attention_capacity=backend['attended'] or None, lanes=[])
        old_lanes = keyed(old['lanes'])
        if set(old_lanes) != set(keyed(backend['lanes'])):
            raise ValueError('lane topology changed within worker instance')
        for lane in backend['lanes']:
            previous = old_lanes[lane['id']]
            durations = {key: difference(lane['time_seconds'][key], previous['time_seconds'][key]) for key in LANE_STATES}
            wall = sum(durations.values())
            rows = difference(lane['rows'], previous['rows'])
            capacity = difference(lane['capacity_rows'], previous['capacity_rows'])
            converted['lanes'].append(dict(id=lane['id'],
                time_fractions={key: value / wall if wall else None for key, value in durations.items()},
                occupied_fraction=1 - durations['idle'] / wall if wall else None,
                evaluations_per_second=rows / wall if wall else None,
                batches_per_second=difference(lane['batches'], previous['batches']) / wall if wall else None,
                batch_fill_fraction=rows / capacity if capacity else None))
        result['backends'].append(converted)
    result['fallback'] = dict(share_fraction=(oversized + retries) / total_requests if total_requests and oversized + retries <= total_requests else None,
                             oversized_per_second=oversized / elapsed, precision_retries_per_second=retries / elapsed)
    return result, max(0, gaps)


class RuntimeCollector:
    def __init__(self, out):
        self.out = Path(out)
        self.samples = deque(maxlen=64)
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.stage, self.error = None, None
        self.last_stat = None

    def sample(self):
        progress = read_object(self.out / 'training/status.json')
        stage = (progress.get('pid'), progress.get('stage'), progress.get('directory'))
        with self.lock:
            if stage != self.stage:
                self.samples.clear(); self.last_stat = None; self.error = None; self.stage = stage
        if progress.get('phase') != 'play' or not progress.get('directory'):
            return
        path = Path(progress['directory']) / 'runtime.json'
        try:
            stat = path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if signature == self.last_stat:
                return
            if stat.st_size > 1024 * 1024:
                raise ValueError('runtime snapshot exceeds size limit')
            raw = validate_snapshot(json.loads(path.read_bytes()))
            if progress.get('child_pid') and raw['pid'] != progress['child_pid']:
                raise ValueError('runtime snapshot belongs to another worker')
            self.accept(raw)
            self.last_stat = signature
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            with self.lock:
                self.error = str(error)

    def accept(self, raw):
        raw = copy.deepcopy(validate_snapshot(raw))
        with self.lock:
            if self.samples and raw['worker_instance_id'] != self.samples[-1]['worker_instance_id']:
                self.samples.clear()
            if self.samples:
                previous = self.samples[-1]
                if raw['sequence'] == previous['sequence']:
                    return
                if raw['sequence'] < previous['sequence'] or raw['elapsed_seconds'] <= previous['elapsed_seconds']:
                    raise ValueError('non-increasing runtime sequence or clock')
                try:
                    aggregate(previous, raw, [previous, raw], {})
                except (ValueError, KeyError):
                    self.samples.clear()
                    self.error = 'runtime counters or topology reset'
            self.samples.append(raw)
            if len(self.samples) > 1:
                self.error = None

    def view(self, context, config, window_seconds=5, now=None):
        now = time.time() if now is None else now
        with self.lock:
            samples, error, stage = list(self.samples), self.error, self.stage
        expected_stage = (context.get('pid'), context.get('stage'), context.get('directory'))
        if stage is not None and stage != expected_stage:
            samples = []
        if not samples:
            return dict(status='unavailable', captured_at=None, worker_instance_id=None, sequence=None,
                        window=dict(requested_seconds=window_seconds, observed_seconds=0, samples=0, missing_samples=0),
                        reason=error or 'worker_telemetry_unavailable', data=None)
        last = samples[-1]
        target = last['elapsed_seconds'] - window_seconds
        first = min(samples, key=lambda s: abs(s['elapsed_seconds'] - target))
        selected = [s for s in samples if s['sequence'] >= first['sequence']]
        observed = last['elapsed_seconds'] - first['elapsed_seconds']
        stale = now - last['captured_at'] > 5
        status, reason, data, gaps = 'warming', 'collecting_window', None, 0
        if len(selected) >= 2 and observed > 0:
            data, gaps = aggregate(first, last, selected, config)
            status, reason = ('partial', 'missed_snapshots') if gaps else ('live', None)
            if observed + .1 < window_seconds:
                status, reason = 'warming', 'collecting_window'
        if error:
            status, reason = 'partial', error
        if stale:
            status, reason = 'stale', 'worker_snapshot_stale'
        return dict(status=status, captured_at=last['captured_at'], worker_instance_id=last['worker_instance_id'],
                    sequence=last['sequence'], reason=reason, data=data,
                    window=dict(requested_seconds=window_seconds, observed_seconds=observed,
                                samples=len(selected), missing_samples=gaps))

    def run(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as error:
                with self.lock:
                    self.error = str(error)
            self.stop.wait(.2)
