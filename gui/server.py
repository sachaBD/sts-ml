"""Local web GUI for the pre-combat outcome model.  .venv/bin/python gui/server.py [--port 8732]"""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model import GROUPS, Ensemble, calibration, natural_dev, presets, tables

HERE = Path(__file__).resolve().parent
VIS = HERE.parents[1] / "sts_visualiser"

# Everything is loaded once at startup.
ens = Ensemble()
t = tables()
dev = natural_dev()
META = dict(
    cards=[c for c in t["cards"] if c["color"] in ("red", "colorless", "curse") or c["type"] == "status"],
    relics=t["relics"], potions=t["potions"], centers=ens.centers,
    groups={g: [e for e in es if e in ens.encounters] for g, es in GROUPS.items()},
    presets=presets(dev), calibration=calibration(ens, dev), seeds=len(ens.nets))


def predict(q):
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

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/meta":
            return self._json(META)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            return self.send_error(404)
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            self._json(predict(q))
        except Exception as e:  # bad edits (e.g. max_hp 0) should not kill the server
            self.send_error(400, str(e))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
