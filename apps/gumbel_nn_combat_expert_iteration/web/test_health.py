"""Health statistics preserve phase boundaries, missingness, and counter semantics."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .health import Health, Lines, Machine, distribution, inference_delta


class DistributionTests(unittest.TestCase):
    def test_histogram_conserves_finite_observations(self):
        for values in ([], [0], [7] * 100, list(range(100)), [-3, 0, 2, math.nan, math.inf]):
            result = distribution(values, 5)
            finite = [v for v in values if math.isfinite(v)]
            self.assertEqual(sum(result['counts']), len(finite))
            self.assertEqual(result['n'], len(finite))
            if finite:
                self.assertLessEqual(result['p10'], result['median'])
                self.assertLessEqual(result['median'], result['p90'])
                self.assertTrue(0 <= result['percentile'] <= 100)
            else:
                self.assertIsNone(result['median'])

    def test_percentile_does_not_add_current_to_reference(self):
        low, high = distribution([1, 2, 3], 0), distribution([1, 2, 3], 100)
        self.assertEqual(low['counts'], high['counts'])
        self.assertEqual(low['percentile'], 0)
        self.assertEqual(high['percentile'], 100)


class IncrementalLogTests(unittest.TestCase):
    def test_partial_lines_and_rotation_are_not_lost_or_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log'
            reader = Lines(limit=8)
            path.write_bytes(b'one\ntwo')
            self.assertEqual(reader.read(path), ['one'])
            self.assertEqual(reader.read(path), [])
            with path.open('ab') as stream:
                stream.write(b'\nthree\n')
            self.assertEqual(reader.read(path), ['two'])
            self.assertTrue(reader.backlog)
            self.assertEqual(reader.read(path), ['three'])
            self.assertFalse(reader.backlog)
            replacement = path.with_suffix('.new')
            replacement.write_text('new\n')
            replacement.replace(path)
            self.assertEqual(reader.read(path), ['new'])
            self.assertEqual(reader.read(path), [])


class NativeMetricTests(unittest.TestCase):
    def test_scaling_counters_and_time_preserves_rates(self):
        previous = dict(seconds=10, leaves=100, cpu_leaves=20, cpu_precision_retries=3, capacity=128)
        current = dict(seconds=20, leaves=300, cpu_leaves=40, cpu_precision_retries=5, capacity=384)
        expected = inference_delta(previous, current)
        self.assertEqual(expected['nn_s'], 22)
        self.assertEqual(expected['batch_fill'], 100 * 200 / 256)
        self.assertEqual(expected['fallback'], 10)
        for scale in (2, 10, 100):
            self.assertEqual(inference_delta({k: v * scale for k, v in previous.items()},
                                            {k: v * scale for k, v in current.items()}), expected)

    def test_missing_reset_and_zero_time_counters_are_not_zero_rates(self):
        self.assertEqual(inference_delta({}, {'seconds': 10}), {})
        self.assertEqual(inference_delta({'seconds': 10}, {'seconds': 10}), {})
        self.assertEqual(inference_delta({'seconds': 10, 'leaves': 5}, {'seconds': 11, 'leaves': 1}), {})
        self.assertEqual(inference_delta({'seconds': 1}, {'seconds': 2}), {})


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        (self.out / 'training').mkdir()
        (self.out / 'inputs').mkdir()
        (self.out / 'inputs/starts.json').write_text('[{}, {}]')
        (self.out / 'config.toml').write_text('[run]\nid="test"\n')
        self.health = Health(self.out)
        self.health.machine.sample = lambda pids, now: ({'gpu': now % 100}, {'cores': []})
        self.clock = patch('apps.gumbel_nn_combat_expert_iteration.web.health.time.time', return_value=100)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        active = patch('apps.gumbel_nn_combat_expert_iteration.web.health.running', return_value=True)
        active.start()
        self.addCleanup(active.stop)
        self.stage('iteration 1', 'play')

    def stage(self, stage, phase):
        self.stage_path = self.out / 'training' / stage.replace(' ', '_')
        self.stage_path.mkdir(exist_ok=True)
        (self.out / 'training/status.json').write_text(json.dumps(dict(
            stage=stage, phase=phase, directory=str(self.stage_path), pid=123, child_pid=456, time=100)))

    def sample(self, now):
        self.now.return_value = now
        self.health.sample()
        return self.health.view()

    def test_current_is_excluded_and_other_phases_do_not_pollute_reference(self):
        self.sample(100)
        self.sample(102)
        self.assertEqual(self.health.view()['metrics']['gpu']['reference']['n'], 1)
        self.stage('test iteration 1', 'play')
        other = self.sample(104)
        self.assertEqual(other['cohort'], 'evaluation')
        self.assertEqual(other['metrics']['gpu']['reference']['n'], 0)
        self.stage('iteration 2', 'play')
        again = self.sample(106)
        self.assertEqual(again['metrics']['gpu']['reference']['n'], 2)
        self.assertEqual(again['metrics']['gpu']['current'], 6)

    def test_references_reset_on_config_change(self):
        self.sample(100)
        (self.out / 'config.toml').write_text('[run]\nid="changed"\n')
        result = self.sample(102)
        self.assertEqual(result['metrics']['gpu']['reference']['n'], 0)

    def test_native_reports_are_not_repeated_observations_or_carried_to_new_stage(self):
        def report(seconds, leaves):
            row = dict(seconds=seconds, leaves=leaves, cpu_leaves=0, cpu_precision_retries=0, capacity=leaves)
            with (self.stage_path / 'worker.log').open('a') as stream:
                stream.write('gpu_stats: ' + json.dumps(row) + '\n')
        report(60, 600)
        self.assertIsNone(self.sample(100)['metrics']['nn_s']['current'])
        report(120, 1200)
        self.assertEqual(self.sample(102)['metrics']['nn_s']['current'], 10)
        repeated = self.sample(104)['metrics']['nn_s']
        self.assertEqual(len(repeated['values']), 1)
        self.assertEqual(repeated['age'], 2)
        self.now.return_value = 240
        self.assertTrue(self.health.view()['metrics']['nn_s']['stale'])
        self.stage('iteration 2', 'play')
        self.assertIsNone(self.sample(242)['metrics']['nn_s']['current'])

    def test_completed_games_and_decisions_are_incremental(self):
        path = self.stage_path / 'results.jsonl'
        row = dict(fight_id='ns:1:deck:0', decisions=7, seconds=3, finished_unix=100)
        path.write_text(json.dumps(row) + '\n')
        first = self.sample(100)
        self.assertEqual(first['games'], 1)
        self.assertEqual(first['actions'], 7)
        self.assertIsNone(first['metrics']['games_s']['current'])
        with path.open('a') as stream:
            stream.write(json.dumps(dict(row, fight_id='ns:1:other:1')) + '\n')
        second = self.sample(102)
        self.assertEqual(second['games'], 2)
        self.assertEqual(second['actions'], 14)
        self.assertEqual(second['decks_seen'], 2)
        self.assertEqual(second['metrics']['games_s']['current'], .5)
        self.assertEqual(second['metrics']['decisions_s']['current'], 3.5)
        self.assertEqual(self.sample(104)['games'], 2)

    def test_history_and_events_have_fixed_bounds(self):
        for i in range(300):
            self.health.event('info', str(i), 'test', i)
        self.assertEqual(len(self.health.events), 200)
        self.assertEqual(self.health.events[0]['text'], '100')
        self.sample(100)
        for i in range(1000):
            self.health.samples.append(dict(self.health.samples[-1], time=100 + i))
        self.assertEqual(len(self.health.samples), self.health.samples.maxlen)


class MockBackendTests(unittest.TestCase):
    def test_mock_rates_preserve_request_accounting(self):
        from .health_mock import MockHealth
        for scenario in ('normal', 'fallback'):
            data = MockHealth(scenario).view()
            metric = lambda key: data['metrics'][key]['current']
            self.assertAlmostEqual(metric('nn_s'), metric('gpu_s') + metric('cpu_oversized_s'))
            self.assertAlmostEqual(metric('fallback') * metric('nn_s') / 100,
                                   metric('cpu_oversized_s') + metric('precision_retry_s'))
            self.assertAlmostEqual(data['native_interval']['requests'], metric('nn_s') * 60)
            self.assertTrue(data['mock'])

    def test_mock_stale_and_missing_are_not_zero_fallback(self):
        from .health_mock import MockHealth
        self.assertTrue(MockHealth('stale').view()['metrics']['fallback']['stale'])
        self.assertNotIn('fallback', MockHealth('missing').view()['metrics'])


class MachineTests(unittest.TestCase):
    def test_absent_sources_are_unknown_not_idle(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('apps.gumbel_nn_combat_expert_iteration.web.health.subprocess.run', side_effect=FileNotFoundError):
            values, details = Machine(Path(directory)).sample([123], 1)
        self.assertEqual(values, {})
        self.assertEqual(details['processes'], [])


if __name__ == '__main__':
    unittest.main()
