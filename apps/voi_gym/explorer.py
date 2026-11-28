import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from math import isfinite
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from agents.combat.search.voi_gym.model import Belief, adaptive_belief, buy
from agents.combat.search.voi_gym.root import (
    DEFAULT_INITIAL_SD_CAP, DEFAULT_PRIOR_SCALE, DEFAULT_PROBE_SD, DEFAULT_SHORTLIST,
)

from .session import Session


def replay(seed, budget, price, prior_scale=DEFAULT_PRIOR_SCALE, initial_sd_cap=DEFAULT_INITIAL_SD_CAP,
           *, belief_factory=Belief, shortlist=DEFAULT_SHORTLIST, probe_sd=DEFAULT_PROBE_SD):
    if not (0 <= seed < 2**32 and 0 <= budget <= 64 and isfinite(price) and price >= 0
            and 0 <= prior_scale <= 1 and 0 <= initial_sd_cap <= .5
            and 1 <= shortlist <= 40 and 0 < probe_sd <= 1):
        raise ValueError('Invalid replay settings')
    session = Session(seed=seed, budget=budget, price=price, prior_scale=prior_scale,
                      initial_sd_cap=initial_sd_cap, belief_factory=belief_factory,
                      shortlist=shortlist, probe_sd=probe_sd)
    frames = []
    observation = None
    while True:
        rows = session.candidates()
        selected = buy(rows)
        frames.append(dict(state=session.snapshot(), rows=rows, selected=selected, observation=observation))
        if selected is None:
            return dict(frames=frames)
        observation = session.step()


def catalog(fights):
    return [dict(name=fight['name'], id=fight['id'], win=fight['win'],
                 decisions=[dict(turn=d['turn'], step=d['step'], probes=len(d['frames'])-1)
                            for d in fight['decisions']]) for fight in fights]


class Explorer(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        self.send(status, 'application/json', json.dumps(data, allow_nan=False).encode())

    def send(self, status, kind, body):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == '/':
            self.send(200, 'text/html; charset=utf-8', Path(__file__).with_suffix('.html').read_bytes())
        elif url.path == '/api/fights':
            self.send_json(catalog(self.server.fights))
        elif url.path == '/api/replay':
            params = parse_qs(url.query, keep_blank_values=True)
            try:
                if 'fight' in params:
                    fight = int(params['fight'][0])
                    decision = int(params.get('decision', ['0'])[0])
                    if fight < 0 or decision < 0:
                        raise ValueError('Invalid recording index')
                    recording = self.server.fights[fight]['decisions'][decision]
                    self.send_json(dict(native=True, **recording))
                    return
                mode = params.get('uncertainty', ['fixed'])[0]
                if mode not in ('fixed', 'adaptive'):
                    raise ValueError('Invalid uncertainty mode')
                data = replay(int(params.get('seed', ['4'])[0]),
                              int(params.get('budget', ['4'])[0]),
                              float(params.get('price', ['.001'])[0]),
                              float(params.get('prior_scale', [DEFAULT_PRIOR_SCALE])[0]),
                              float(params.get('initial_sd_cap', [DEFAULT_INITIAL_SD_CAP])[0]),
                              belief_factory=adaptive_belief if mode == 'adaptive' else Belief,
                              shortlist=int(params.get('shortlist', [DEFAULT_SHORTLIST])[0]),
                              probe_sd=float(params.get('probe_sd', [DEFAULT_PROBE_SD])[0]))
                self.send_json(data)
            except (ValueError, IndexError):
                self.send_json(dict(error='Invalid replay settings'), status=400)
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description='Serve the measurement VOI explorer')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8006)
    parser.add_argument('--recordings', type=Path)
    args = parser.parse_args()
    fights = []
    if args.recordings:
        with args.recordings.open() as rows:
            fights = [json.loads(row) for row in rows]
        # Recordings made before these knobs were exposed used the original constants.
        for fight in fights:
            for decision in fight['decisions']:
                for frame in decision['frames']:
                    frame['state'].setdefault('shortlist', 4)
                    frame['state'].setdefault('probe_sd', .08)
    with ThreadingHTTPServer((args.host, args.port), Explorer) as server:
        server.fights = fights
        print(f'VOI explorer: http://{args.host}:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
