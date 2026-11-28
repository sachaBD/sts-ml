import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from math import isfinite
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import parse_qs, urlsplit

from agents.combat.search.voi_gym.model import Belief, adaptive_belief, buy
from agents.combat.search.voi_gym.root import (
    DEFAULT_INITIAL_SD_CAP, DEFAULT_PRIOR_SCALE, DEFAULT_PROBE_SD, DEFAULT_SHORTLIST,
)

from .live_combat import bridge_process, play_branch, settings
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


def catalog(fights, starts=None):
    return [dict(name=fight['name'], id=fight['id'], win=fight['win'],
                 live=starts is not None and fight['id'] in starts,
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
            self.send_json(catalog(self.server.fights, self.server.starts if self.server.bridge else None))
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


    def do_POST(self):
        if self.path != '/api/live' or not self.server.bridge:
            self.send_error(404)
            return
        if not self.server.live_slots.acquire(blocking=False):
            self.send_json(dict(error='Live replay is busy; try again.'), status=503)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 128_000 or self.headers.get_content_type() != 'application/json':
                raise ValueError('Invalid live replay request')
            body = json.loads(self.rfile.read(size))
            fight, origin = int(body['fight']), int(body['decision'])
            operation = body['operation']
            if fight < 0 or origin < 0 or operation not in ('rerun', 'action', 'turn'):
                raise ValueError('Invalid live replay request')
            recording = self.server.fights[fight]
            decisions = recording['decisions']
            if origin >= len(decisions) or len(body['history']) > 512:
                raise ValueError('Invalid live replay position')
            start = self.server.starts[recording['id']]
            history = [settings(entry) for entry in body['history']]
            config = settings(body['settings'])
            with bridge_process(self.server.bridge) as proc:
                data = play_branch(proc, start, decisions, origin, history, config, operation)
            self.send_json(data)
        except (ValueError, TypeError, IndexError, KeyError, OverflowError) as error:
            self.send_json(dict(error=str(error)), status=400)
        except (RuntimeError, OSError) as error:
            self.send_json(dict(error=str(error)), status=502)
        finally:
            self.server.live_slots.release()


def main():
    parser = argparse.ArgumentParser(description='Serve the measurement VOI explorer')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8006)
    parser.add_argument('--recordings', type=Path)
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--starts', type=Path)
    args = parser.parse_args()
    live = (args.bridge, args.model, args.starts)
    if any(live) and (not all(live) or not args.recordings):
        parser.error('live replay requires --bridge, --model, --starts, and --recordings')
    starts = {}
    if args.starts:
        with args.starts.open() as rows:
            for line in rows:
                row = json.loads(line)
                starts[row['fight_id']] = row['start']
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
        server.starts = starts
        server.bridge = [args.bridge.resolve(), args.model.resolve(), '--trace'] if args.bridge else None
        server.live_slots = BoundedSemaphore(2)
        print(f'VOI explorer: http://{args.host}:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
