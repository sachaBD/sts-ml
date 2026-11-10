"""Queue telemetry names each backend and accounts for every request it evaluated."""
import json
from pathlib import Path
import tempfile
import unittest

from infra.workers.telemetry import queues


def record(time, requests):
    return dict(time=time, actors=[], lanes=[], queues=[
        dict(backend=name, requests=n, waited_seconds=0.001 * n, depth=0, peak_depth=3) for name, n in requests.items()])


class QueueTelemetry(unittest.TestCase):
    def test_each_queue_reports_its_backend_total_and_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'telemetry.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in (
                record(0, {'gpu-48': 0, 'gpu-128': 0, 'cpu': 0}),
                record(10, {'gpu-48': 9_000, 'gpu-128': 990, 'cpu': 10}))))
            states = queues(path)
        self.assertEqual([s.backend for s in states], ['gpu-48', 'gpu-128', 'cpu'])
        self.assertEqual([s.requests for s in states], [9_000, 990, 10])
        self.assertAlmostEqual(sum(s.share_pct for s in states), 100)
        self.assertAlmostEqual(states[-1].share_pct, 0.1)
        self.assertEqual([s.requests_s for s in states], [900, 99, 1])


if __name__ == '__main__':
    unittest.main()
