"""Champ viewer: replay any recorded Champ fight from a combat_v4 run, step / play it on a timeline, see the PV net's
value and priors, and take over or edit the board at any point.

  .venv/bin/python gui/champ_server.py [--port 8733]     # http://127.0.0.1:8733/champ/
  link to a fight: /champ/?date=2026-10-04&id=champ-ox-c-r01&part=bench-oracle&fight=<fight_id>

A fight is (dataset, fight_id), where a dataset is a directory of fights-*.parquet under runs/schema=combat_v4/
(date=…/id=…[/part]). The browser never holds a start (its RNG seeds are uint64): it sends {base, patch, ops} and this
server rebuilds the battle in build/champ-viewer/champ_session (apps/champ_viewer/session.cpp) for every request.
"""
import argparse
import copy
import json
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb

from model import tables

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIS = ROOT.parent / "sts_visualiser"
SESSION = ROOT / "build/champ-viewer/champ_session"
PV_MODEL = ROOT / "runs/schema=combat_v4/date=2026-10-05/id=champ-diag-D5/model/model.onnx"
RUNS = ROOT / "runs/schema=combat_v4"
CHAMP_ENCOUNTER = 39
CHAMP_MOVES = ["THE_CHAMP_HEAVY_SLASH", "THE_CHAMP_FACE_SLAP", "THE_CHAMP_EXECUTE", "THE_CHAMP_DEFENSIVE_STANCE",
               "THE_CHAMP_GLOAT", "THE_CHAMP_TAUNT", "THE_CHAMP_ANGER"]
PATCHABLE = {"deck", "relics", "potions", "potion_capacity", "hp", "max_hp", "seed"}


def datasets():
    """Every directory holding fights-*.parquet, as {date, id, part}."""
    out = []
    for d in sorted({p.parent for p in RUNS.glob("date=*/id=*/**/fights-*.parquet")}):
        rel = d.relative_to(RUNS).parts
        out.append(dict(date=rel[0].removeprefix("date="), id=rel[1].removeprefix("id="), part="/".join(rel[2:])))
    return out


def dataset_dir(date, id, part=""):
    d = (RUNS / f"date={date}" / f"id={id}" / part).resolve()
    if RUNS.resolve() not in d.parents or not list(d.glob("fights-*.parquet")):
        raise ValueError(f"no fights in {d}")
    return d


def sql(q, *args):
    with DB_LOCK:
        return DB.execute(q, list(args)).fetchall()


DB, DB_LOCK = duckdb.connect(), threading.Lock()
DB.execute("set memory_limit = '1GB'; set threads = 2")
T = tables()
META = dict(cards=T["cards"], relics=T["relics"], potions=T["potions"], moves=CHAMP_MOVES,
            pv_model=str(PV_MODEL.relative_to(ROOT)))


def meta(q):
    return dict(META, datasets=datasets())  # rescanned per page load (~30 ms): new runs show up without a restart
STARTS = {}  # base key "date|id|part|fight_id" -> start, filled when a fight is loaded


def fights(q):
    d = dataset_dir(q["date"], q["id"], q.get("part", ""))
    rows = sql(f"""select fight_id, won, final_hp, len(actions) from '{d}/fights-*.parquet'
                   where start.encounter = {CHAMP_ENCOUNTER} order by fight_id""")
    return dict(fights=[dict(fight_id=f, won=w, final_hp=hp, n=n) for f, w, hp, n in rows])


def fight(q):
    d = dataset_dir(q["date"], q["id"], q.get("part", ""))
    rows = sql(f"""select to_json(start)::varchar, actions, won, final_hp, agent from '{d}/fights-*.parquet'
                   where fight_id = ?""", q["fight"])
    if not rows:
        raise ValueError(f"no fight {q['fight']}")
    start, actions, won, final_hp, agent = rows[0]
    start = json.loads(start)
    base = "|".join([q["date"], q["id"], q.get("part", ""), q["fight"]])
    STARTS[base] = start
    view = dict(hp=start["hp"], max_hp=start["max_hp"], seed=start["seed"],
                deck=[dict(id=c["id"], upgraded=int(c["upgraded"]), misc=c["misc"]) for c in start["deck"]],
                relics=[dict(id=r["id"], data=r["data"]) for r in start["relics"]], potions=start["potions"])
    return dict(base=base, start=view, actions=actions, won=won, final_hp=final_hp, agent=agent)


class Session:
    """One champ_session process: a request line in, a response line out."""

    def __init__(self, model=None):
        args = [str(SESSION)] + ([str(model)] if model else [])
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.lock = threading.Lock()

    def ask(self, request):
        with self.lock:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"champ_session died (exit {self.proc.poll()})")
        return json.loads(line)


FAST = Session(PV_MODEL if PV_MODEL.exists() else None)


def query(q):
    """The request's start (recorded start + the browser's pre-battle patch) and ops, rebuilt and viewed."""
    if q["base"] not in STARTS:  # server restarted under an open page
        date, id, part, fid = q["base"].split("|", 3)
        fight(dict(date=date, id=id, part=part, fight=fid))
    start = copy.deepcopy(STARTS[q["base"]])
    patch = q.get("patch") or {}
    if set(patch) - PATCHABLE:
        raise ValueError(f"cannot patch {set(patch) - PATCHABLE}")
    start.update(patch)
    if "deck" in patch:  # bottled-card indices refer to the original deck
        start["deck"] = [dict(id=int(c["id"]), upgraded=bool(c["upgraded"]), misc=int(c.get("misc", 0))) for c in patch["deck"]]
        start["bottled"] = [-1, -1, -1]
    if "potions" in patch:
        start["potion_capacity"] = len(start["potions"])
    kind = q.get("query", "pv")
    if kind not in ("view", "pv"):
        raise ValueError(f"unknown query {kind}")
    return FAST.ask(dict(start=start, ops=q.get("ops", []), query=kind))  # query_error (no PV when done) keeps the view


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE / "web"), **k)

    def translate_path(self, path):
        if path.startswith("/vis/"):
            return str(VIS / path[5:].split("?")[0])
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
        routes = {"/api/champ/meta": meta, "/api/champ/fights": fights, "/api/champ/fight": fight}
        if url.path not in routes:
            return super().do_GET()
        try:
            self._json(routes[url.path]({k: v[0] for k, v in parse_qs(url.query).items()}))
        except Exception as e:
            self._json(dict(error=f"{type(e).__name__}: {e}"), 400)

    def do_POST(self):
        if self.path != "/api/champ/query":
            return self.send_error(404)
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            self._json(query(q))
        except Exception as e:  # a bad edit must not kill the server
            self._json(dict(error=f"{type(e).__name__}: {e}"), 400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8733)
    a = ap.parse_args()
    print(f"champ viewer: http://127.0.0.1:{a.port}/champ/  ({len(datasets())} datasets, "
          f"PV model {'on' if PV_MODEL.exists() else 'off'})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
