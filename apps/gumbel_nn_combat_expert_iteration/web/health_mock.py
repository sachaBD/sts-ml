"""Moving UI demo. Synthetic component snapshots; no worker or GPU required."""
import math
import time

SCENARIOS = ('normal', 'fallback', 'stale', 'missing')


class MockHealth:
    def __init__(self, scenario='normal'):
        if scenario not in SCENARIOS:
            raise ValueError(f'unknown mock scenario: {scenario}')
        self.scenario, self.started = scenario, time.time()

    def _instant(self, now):
        elapsed = max(0, now - self.started)
        cycle = elapsed % 48
        mode = 'Balanced' if cycle < 12 else 'Queueing' if cycle < 24 else 'GPU starved' if cycle < 36 else 'Draining'
        actors = []
        for i in range(9):
            wave = (1 + math.sin(elapsed * .8 + i * .9)) / 2
            empty = int(224 * max(0, (cycle - 36) / 12 - i * .045)) if mode == 'Draining' else 0
            active = 224 - empty
            ready_fraction = (.04 + wave * .12) if mode == 'Queueing' else (.55 + wave * .3) if mode == 'GPU starved' else .15 + wave * .45
            ready = int(active * ready_fraction)
            working = (.12 + wave * .18) if mode == 'Queueing' else (.7 + wave * .25) if mode == 'GPU starved' else .3 + wave * .5
            phase = 'done' if not active else 'working' if (elapsed * .65 + i * .13) % 1 < working else 'waiting'
            actors.append(dict(id=i, ready=ready, waiting=active - ready, empty=empty,
                               phase=phase, working_fraction=working if active else 0,
                               completed=int(elapsed * (5 + i * .4))))
        queued = int(600 + 180 * math.sin(elapsed)) if mode == 'Queueing' else int(8 + 6 * math.sin(elapsed * 1.4)) if mode == 'GPU starved' else int(110 + 80 * math.sin(elapsed * .7))
        if mode == 'Draining':
            queued = int(queued * (48 - cycle) / 12)
        dispatchers, lanes = [], []
        for i in range(2):
            beat = (elapsed * .7 + i * .43) % 1
            occupancy = .97 if mode == 'Queueing' else .28 if mode == 'GPU starved' else .65 if mode == 'Draining' else .82
            active = beat < occupancy
            progress = beat / occupancy if active else 0
            phase = 'idle' if not active else 'packing' if progress < .12 else 'h2d' if progress < .25 else 'inference' if progress < .8 else 'd2h'
            rows = min(512, int(420 + 80 * math.sin(elapsed * 1.3 + i))) if mode == 'Queueing' else int(20 + 70 * (1 + math.sin(elapsed + i)) / 2) if mode == 'GPU starved' else int(160 + 330 * (1 + math.sin(elapsed + i)) / 2)
            capacity = next(n for n in (32, 256, 512) if n >= rows)
            lanes.append(dict(id=i, phase=phase, rows=rows,
                              capacity=capacity, occupied_fraction=occupancy,
                              batches=int(elapsed * (25 + i * 2))))
            dispatchers.append(dict(id=i, phase='evaluating' if active else 'gathering' if queued else 'idle',
                                    rows=rows if active else min(queued, 512)))
        return dict(version=1, time=now, window_seconds=2, mode=mode, actors=actors,
                    queue=dict(depth=queued, capacity=9 * 224,
                               oldest_ms=(queued / 35 if mode == 'Queueing' else queued / 100),
                               wait_p50_ms=queued / 140, wait_p95_ms=queued / 40, samples=512),
                    dispatchers=dispatchers, lanes=lanes)

    def topology(self, now):
        samples = [self._instant(now - 5 + i * .2) for i in range(26)]
        average = lambda values: sum(values) / len(values)
        lanes, dispatchers = [], []
        for i in range(2):
            observations = [sample['lanes'][i] for sample in samples]
            batch_rates = [(25 + i * 2) * lane['occupied_fraction'] for lane in observations]
            eval_rates = [rate * lane['rows'] for rate, lane in zip(batch_rates, observations)]
            capacity_rates = [rate * lane['capacity'] for rate, lane in zip(batch_rates, observations)]
            occupied = average([lane['occupied_fraction'] for lane in observations])
            batches_s, evals_s = average(batch_rates), average(eval_rates)
            fill = sum(eval_rates) / sum(capacity_rates)
            fractions = {name: weight * occupied for name, weight in
                         [('packing', .12), ('h2d', .13), ('inference', .55), ('d2h', .2)]}
            fractions['idle'] = 1 - occupied
            lanes.append(dict(id=i, occupied_fraction=occupied, time_fractions=fractions,
                              batches_s=batches_s, evals_s=evals_s, batch_fill=fill))
            dispatchers.append(dict(id=i, batches_s=batches_s, requests_s=evals_s,
                                    batch_fill=fill))
        depths = [sample['queue']['depth'] for sample in samples]
        arrival_rate = max(0, sum(lane['evals_s'] for lane in lanes) + (depths[-1] - depths[0]) / 5)
        actors = []
        weights = [1 + .2 * math.sin(max(0, now - self.started) * .12 + i) for i in range(9)]
        for i in range(9):
            actor = dict(id=i, requests_s=arrival_rate * weights[i] / sum(weights))
            for key in ('ready', 'waiting', 'empty', 'working_fraction'):
                actor[key] = average([sample['actors'][i][key] for sample in samples])
            actors.append(actor)
        waits = sorted(-math.log(1 - (i + .5) / 16) * sample['queue']['depth'] / 140
                       for sample in samples for i in range(16))
        return dict(version=1, time=now, window_seconds=5, mode=samples[-1]['mode'], actors=actors,
                    queue=dict(depth=average(depths), peak=max(depths), capacity=9 * 224,
                               wait_p50_ms=waits[len(waits) // 2],
                               wait_p95_ms=waits[int(.95 * (len(waits) - 1))], samples=len(waits)),
                    dispatchers=dispatchers, lanes=lanes)

    def view(self):
        now = time.time()
        elapsed = max(0, now - self.started)
        topology = self.topology(now)
        oversized, retries = (4000, 2000) if self.scenario == 'fallback' else (30, 15)
        requests = 60000 + oversized
        native_values = dict(nn_s=requests / 60, gpu_s=1000, cpu_oversized_s=oversized / 60,
                             precision_retry_s=retries / 60, fallback=100 * (oversized + retries) / requests,
                             batch_fill=100 * 60000 / 65536)
        age = 300 if self.scenario == 'stale' else 3
        metrics = {key: dict(current=value, age=age, stale=age > 130) for key, value in native_values.items()}
        cpu = 35 + 15 * math.sin(elapsed * .2)
        gpu = 100 * sum(lane['occupied_fraction'] for lane in topology['lanes']) / 2
        for key, value in dict(cpu=cpu, gpu=gpu, vram=6, ram=12, disk_free=40,
                               games_s=70, decisions_s=2100).items():
            metrics[key] = dict(current=value, age=0, stale=False)
        if self.scenario == 'missing':
            metrics = {k: v for k, v in metrics.items() if k not in native_values}
            topology = None
        elif self.scenario == 'stale':
            topology['time'] = now - 300
        return dict(version=1, ready=True, mock=True, time=now, interval=1, state='running',
                    config={'run': {'id': f'MOCK · {self.scenario}'},
                            'workers': dict(actors=9, slots=224, gpu_batch=512,
                                            gpu_bank=[256, 32], gpu_wait_us=100)},
                    progress=dict(stage='iteration 42', phase='play', done=7400 + int(elapsed * 5) % 2600, total=10000),
                    metrics=metrics, topology=topology,
                    machine=dict(cores=[max(0, min(100, 45 + 40 * math.sin(elapsed * .4 + i))) for i in range(12)],
                                 gpu_name='Mock GPU', vram_total=8, ram_total=32, ram_available=20,
                                 disk_total=256, disk_used=210))
