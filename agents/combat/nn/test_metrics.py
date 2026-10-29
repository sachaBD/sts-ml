"""Metric definitions must not conflate useful decisions, NN rows or hardware activity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.combat.nn.metrics import DeviceSampler, device_metrics, last_json, play_metrics, sampled_devices


class MetricsTests(unittest.TestCase):
    def test_decisions_are_not_nn_evaluations(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            log='\n'.join([
                'play_stats: '+json.dumps(dict(seconds=2,games=10,searched_decisions=100,wall_start_unix=1000)),
                'actors_stats: '+json.dumps(dict(actor_cpu_pct=25,server_cpu_pct=40,requests=10000)),
                'gpu_stats: '+json.dumps(dict(leaves=10000,capacity=20000)),
                'gpu_stats: {"leaves":'])
            (p/'worker.log').write_text(log)
            m=play_metrics(p)
            self.assertEqual(m['decisions_per_second'],50)
            self.assertEqual(m['nn_evaluations_per_second'],5000)
            self.assertEqual(m['games_per_hour_gameplay'],18000)
            self.assertEqual(m['batch_fill_pct'],50)
            self.assertEqual(m['worker_cpu_pct'],25)
            self.assertIsNone(m['gpu_compute_throughput_pct'])
            self.assertIsNone(m['coordinator_io_bound_pct'])

    def test_cpu_fallback_counts_as_real_inference(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            (p/'worker.log').write_text('play_stats: {"seconds":2,"games":1,"searched_decisions":1}\n'
                                       'actors_stats: {"requests":100}\n'
                                       'gpu_stats: {"leaves":80,"cpu_leaves":20,"capacity":128}\n')
            self.assertEqual(play_metrics(p)['nn_evaluations_per_second'],50)

    def test_gpu_busy_memory_activity_and_peak_are_distinct(self):
        m=device_metrics(dict(gpu_util=dict(mean=99),memory_util=dict(mean=85),vram_mib=dict(mean=6000)),12000)
        self.assertEqual(m['gpu_busy_pct'],99)
        self.assertEqual(m['gpu_vram_pct'],50)
        self.assertEqual(m['gpu_memory_activity_pct'],85)
        self.assertIsNone(m['gpu_memory_bandwidth_pct'])
        self.assertIsNone(m['gpu_compute_throughput_pct'])

    def test_missing_counters_are_not_zero(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(play_metrics(root)['worker_cpu_pct'])
        self.assertEqual(last_json('runtime_stats: {', 'runtime_stats'),{})
        with patch('shutil.which',return_value=None), patch('subprocess.Popen') as popen:
            with DeviceSampler('unused'): pass
            popen.assert_not_called()

    def test_sample_window_and_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'gpu.csv'
            p.write_text('timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W]\n'
                         '2026/10/27 10:00:00.000, 90, 70, 6000, 12000, 100\n')
            from datetime import datetime
            stamp=datetime(2026,10,27,10).timestamp()
            m=sampled_devices(p,stamp-.1,stamp+.1)
            self.assertEqual(m['samples'],1)
            self.assertEqual(m['metrics']['gpu_vram_pct'],50)
            self.assertEqual(sampled_devices(p,stamp+1,stamp+2)['samples'],0)


if __name__=='__main__': unittest.main()
