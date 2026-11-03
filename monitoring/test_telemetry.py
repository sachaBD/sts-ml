"""Queue depth is a latest-sample gauge; peak depth is a worker-lifetime high-water mark."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .telemetry import queues


class QueueDepthTests(unittest.TestCase):
    def read(self, rows):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'telemetry.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            return queues(path)

    def test_depth_and_peak_are_latest_values_not_counter_deltas(self):
        rows = [dict(time=t, queues=[dict(requests=requests, waited_seconds=wait,
                                         depth=depth, peak_depth=peak)])
                for t, requests, wait, depth, peak in [(0, 0, 0, 8, 8), (3, 30, .03, 2, 8), (5, 50, .05, 5, 9)]]
        state, = self.read(rows)
        self.assertEqual(state.depth, 5)
        self.assertEqual(state.peak_depth, 9)
        self.assertEqual(state.requests_s, 10)
        self.assertEqual(state.wait_ms, 1)

    def test_drained_and_unused_queues_keep_distinct_high_water_marks(self):
        rows = [dict(time=t, queues=[dict(requests=10, waited_seconds=.01, depth=depth, peak_depth=4),
                                    dict(requests=0, waited_seconds=0, depth=0, peak_depth=0)])
                for t, depth in [(0, 4), (5, 0)]]
        drained, unused = self.read(rows)
        self.assertEqual((drained.depth, drained.peak_depth), (0, 4))
        self.assertEqual((unused.depth, unused.peak_depth), (0, 0))
        self.assertIsNone(drained.wait_ms)
        self.assertIsNone(unused.wait_ms)

    def test_older_telemetry_reports_unknown_depth_not_zero(self):
        rows = [dict(time=t, queues=[dict(requests=t, waited_seconds=0)]) for t in (0, 5)]
        state, = self.read(rows)
        self.assertIsNone(state.depth)
        self.assertIsNone(state.peak_depth)
        self.assertEqual(state.requests_s, 1)


if __name__ == '__main__':
    unittest.main()
