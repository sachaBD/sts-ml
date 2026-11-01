"""Local server for all GUI pages.  .venv/bin/python gui/server.py [--port 8732]"""
import argparse
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import gumbel_server
from model import Ensemble, calibration, encounter_groups, natural_dev, presets, tables

HERE = Path(__file__).resolve().parent
VIS = HERE.parents[1] / "sts_visualiser"

# Outcome calibration is expensive; load it only when its API is first used.
ens = META = None
MODEL_LOCK = threading.Lock()


def outcome_meta():
    global ens, META
    with MODEL_LOCK:
        if META is None:
            print("Loading outcome model and calibration...", flush=True)
            ens = Ensemble()
            t = tables()
            dev = natural_dev()
            groups = encounter_groups(ens.encounters)
            META = dict(
                cards=[c for c in t["cards"] if c["color"] in ("red", "colorless", "curse") or c["type"] == "status"],
                relics=t["relics"], potions=t["potions"], centers=ens.centers,
                groups=groups, presets=presets(dev, groups), calibration=calibration(ens, dev, groups), seeds=len(ens.nets))
    return META


CHAMP_LOCK = threading.Lock()


def champ_api(name, q):
    # Reuse the standalone Champ backend, initializing it only on first use.
    with CHAMP_LOCK:
        import champ_server
    return getattr(champ_server, name)(q)


def predict(q):
    outcome_meta()
    # One batch: the state, the state plus each candidate card, and the state against every fight.
    enc, pre = q["encounter"], q["pre"]
    rows = [dict(encounter=enc, pre=pre)]
    for c in q.get("candidates", []):
        rows.append(dict(encounter=enc, pre=dict(pre, deck=pre["deck"] + [dict(c, misc=0)])))
    fights = [e for es in META["groups"].values() for e in es]
    rows += [dict(encounter=e, pre=pre) for e in fights]
    s = ens.score(rows)
    n = 1 + len(q.get("candidates", []))
    return dict(base=s[0], candidates=s[1:n], fights=dict(zip(fights, s[n:])))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE / "web"), **k)

    def translate_path(self, path):
        if path.startswith("/vis/"):  # card art + metadata from ../sts_visualiser
            return str(VIS / path[5:].split("?")[0])
        # Generated browser-model data is also available while developing the source pages.
        if path.startswith("/data/"):
            return str(HERE / "site" / path[1:].split("?")[0])
        if path == "/assets/browser-model.js":
            return str(HERE / "browser_model.js")
        if path == "/assets/run-value-model.js":
            return str(HERE / "run_value_model.js")
        return super().translate_path(path)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        routes = {"/api/meta": lambda q: outcome_meta(),
                  "/api/champ/meta": lambda q: champ_api("meta", q),
                  "/api/champ/fights": lambda q: champ_api("fights", q),
                  "/api/champ/fight": lambda q: champ_api("fight", q),
                  "/api/gumbel/runs": gumbel_server.runs, "/api/gumbel/run": gumbel_server.run}
        if url.path not in routes:
            return super().do_GET()
        try:
            self._json(routes[url.path]({k: v[0] for k, v in parse_qs(url.query).items()}))
        except Exception as e:
            self._json(dict(error=f"{type(e).__name__}: {e}"), 400)

    def do_POST(self):
        routes = {"/api/predict": predict, "/api/champ/query": lambda q: champ_api("query", q),
                  "/api/gumbel/decision": gumbel_server.decision}
        if self.path not in routes:
            return self.send_error(404)
        try:
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._json(routes[self.path](q))
        except Exception as e:
            self._json(dict(error=f"{type(e).__name__}: {e}"), 400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", type=Path, action="append", default=[], help="another runs/ root for the Gumbel tree viewer")
    ap.add_argument("--nn-explain", type=Path, default=gumbel_server.EXPLAIN)
    a = ap.parse_args()
    gumbel_server.RUN_ROOTS += a.runs
    gumbel_server.EXPLAIN = a.nn_explain
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Ready: http://{a.host}:{a.port}/index.html", flush=True)
    server.serve_forever()
