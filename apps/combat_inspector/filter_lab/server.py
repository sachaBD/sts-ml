"""Step-through filter lab: serves recorded filter_replay decisions and re-runs one decision with other settings.

    .venv/bin/python -m apps.combat_inspector.filter_lab.server TRACE STARTS MODEL_DIR [--port 8005]

STARTS holds fight starts; TRACE is `filter_replay MODEL_DIR < STARTS`.
"""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import argparse, json, subprocess, sys, time
from apps.combat_inspector.filter_lab import page

REPLAY = Path(__file__).resolve().parents[3] / 'build/main/apps/combat_inspector/filter_replay'
MODES = {'full', 'root_only', 'fixed_uncertainty'}


class Lab:
    def __init__(self, trace, starts, model):
        self.trace, self.model = Path(trace), Path(model)
        self.starts = {json.loads(l)['fight_id']: l for l in Path(starts).read_text().splitlines()}

    def html(self):
        return page.page(page.load_records(self.trace))

    def run(self, q):
        fight, step = q['fight'], int(q['step'])
        s = {'price': float(q['price']), 'cap': int(q['cap']), 'worlds': int(q['worlds']), 'mode': q['mode']}
        if fight not in self.starts or s['mode'] not in MODES or not (0 <= s['cap'] <= 2000 and 1 <= s['worlds'] <= 64 and s['price'] >= 0):
            raise ValueError('bad parameters')
        t = time.perf_counter()
        args = [str(REPLAY), str(self.model), str(step), str(s['cap']), str(s['worlds']), repr(s['price']), s['mode']]
        out = subprocess.run(args, input=self.starts[fight] + '\n', capture_output=True, text=True, timeout=120, check=True).stdout
        d = page.decision(json.loads(out), s)
        d['ms'] = round(1000 * (time.perf_counter() - t))
        return d


def handler(lab):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind):
            data = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ('/', '/index.html'):
                return self.send(200, lab.html(), 'text/html; charset=utf-8')
            if url.path == '/api/run':
                try:
                    body = lab.run({k: v[0] for k, v in parse_qs(url.query).items()})
                    return self.send(200, json.dumps(body, allow_nan=False), 'application/json')
                except (KeyError, ValueError, subprocess.SubprocessError) as e:
                    return self.send(400, json.dumps({'error': str(e)}), 'application/json')
            self.send(404, '{"error":"not found"}', 'application/json')

        def log_message(self, fmt, *args):
            sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % args))
    return Handler


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('trace'); p.add_argument('starts'); p.add_argument('model'); p.add_argument('--port', type=int, default=8005)
    a = p.parse_args()
    ThreadingHTTPServer(('0.0.0.0', a.port), handler(Lab(a.trace, a.starts, a.model))).serve_forever()
