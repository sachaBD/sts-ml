"""The GUI serves its assets without replacing the production telemetry routes."""
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


@contextmanager
def dashboard(*args):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, '-m', 'apps.gumbel_nn_combat_expert_iteration.web',
         *args, '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(base, timeout=1):
                    break
            except URLError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError('Dashboard did not start')
                time.sleep(.05)
        yield base
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


class WebRouteTests(unittest.TestCase):
    def get(self, base, path):
        with urlopen(base + path, timeout=5) as response:
            return response.headers.get_content_type(), response.read()

    def test_mock_assets_and_production_routes_remain_distinct(self):
        with dashboard('--mock', 'normal') as base:
            for path, kind in [('/', 'text/html'), ('/health.js', 'text/javascript'),
                               ('/health.css', 'text/css'), ('/telemetry.js', 'text/javascript')]:
                actual, body = self.get(base, path)
                self.assertEqual(actual, kind)
                self.assertTrue(body)
            _, body = self.get(base, '/api/mock')
            self.assertTrue(json.loads(body)['ready'])
            for component in ('actors', 'queues', 'lanes'):
                _, body = self.get(base, f'/api/{component}')
                self.assertEqual(json.loads(body), [])  # no synthetic production counters
            for path in ('/api/health', '/api/unknown'):
                with self.assertRaises(HTTPError) as error:
                    self.get(base, path)
                self.assertEqual(error.exception.code, 404)

    def test_live_routes_still_read_native_counters(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            stage = out / 'training/iter000'
            stage.mkdir(parents=True)
            status = out / 'training/status.json'
            status.write_text(json.dumps(dict(phase='play', directory=str(stage))))
            records = [dict(time=t,
                            actors=[dict(requests=10*t, blocked_seconds=.5*t)],
                            queues=[dict(requests=10*t, waited_seconds=.1*t)],
                            lanes=[dict(queue=0, requests=10*t, capacity=20*t, busy_seconds=.8*t)])
                       for t in (0, 5)]
            (stage / 'telemetry.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            with dashboard(str(out)) as base:
                for path in ('/api/health', '/api/mock'):
                    with self.assertRaises(HTTPError) as error:
                        self.get(base, path)
                    self.assertEqual(error.exception.code, 404)
                for component in ('actors', 'queues', 'lanes'):
                    _, body = self.get(base, f'/api/{component}')
                    self.assertEqual(json.loads(body)[0]['requests_s'], 10)
                status.write_text(json.dumps(dict(phase='train')))
                for component in ('actors', 'queues', 'lanes'):
                    _, body = self.get(base, f'/api/{component}')
                    self.assertEqual(json.loads(body), [])


if __name__ == '__main__':
    unittest.main()
