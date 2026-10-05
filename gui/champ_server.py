"""Champ viewer: sandbox the Champ fight, see the PV net's priors / value and the teacher's search and playouts.

  .venv/bin/python gui/champ_server.py [--port 8733] [--workers 2]     # http://127.0.0.1:8733/champ/

Starts are the held-out bench2k Champ decks (409 decks × 5 battle seeds) with the teacher's recorded fights on
them. The browser never holds a start (its RNG seeds are uint64): it sends {base fight_id, patch, ops} and this
server rebuilds the battle in build/champ-viewer/champ_session (apps/champ_viewer/session.cpp) for every request.
Cheap queries (view, PV) share one session process; teacher searches / playouts get their own processes,
at most --workers at a time.
"""
import argparse
import copy
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb

from model import tables

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIS = ROOT.parent / "sts_visualiser"
SESSION = ROOT / "build/champ-viewer/champ_session"
PV_MODEL = ROOT / "runs/schema=combat_v4/date=2026-10-05/id=champ-diag-D5/model/model.onnx"
STARTS = ROOT / "runs/schema=pv_starts_v1/id=champ-bench2k/starts.parquet"
TEACHER = ROOT / "runs/schema=combat_v4/date=2026-10-05/id=champ-bench2k-teacher"
RECORD = ROOT / "experiments/champ-web-viewer/user_fights.jsonl"
CHAMP_MOVES = ["THE_CHAMP_HEAVY_SLASH", "THE_CHAMP_FACE_SLAP", "THE_CHAMP_EXECUTE", "THE_CHAMP_DEFENSIVE_STANCE",
               "THE_CHAMP_GLOAT", "THE_CHAMP_TAUNT", "THE_CHAMP_ANGER"]
PATCHABLE = {"deck", "relics", "potions", "potion_capacity", "hp", "max_hp", "seed"}


def load():
    db = duckdb.connect()
    db.execute("set memory_limit = '1GB'; set threads = 2")
    starts = {r[0]: (r[1], json.loads(r[2])) for r in db.execute(
        f"select fight_id, source_fight_id, to_json(start)::varchar from '{STARTS}'").fetchall()}
    teacher = {r[0]: dict(won=r[1], final_hp=r[2], actions=r[3]) for r in db.execute(
        f"select fight_id, won, final_hp, actions from '{TEACHER}/fights-*.parquet'").fetchall()}
    decks = {}
    for fid, (source, start) in sorted(starts.items()):
        d = decks.setdefault(source, dict(source=source, hp=start["hp"], max_hp=start["max_hp"], floor=start["floor"],
                                          deck=[dict(id=c["id"], upgraded=int(c["upgraded"]), misc=c["misc"]) for c in start["deck"]],
                                          relics=[dict(id=r["id"], data=r["data"]) for r in start["relics"]],
                                          potions=start["potions"],
                                          fights=[]))
        t = teacher.get(fid)
        d["fights"].append(dict(fight_id=fid, seed=start["seed"], won=t and t["won"], final_hp=t and t["final_hp"]))
    for d in decks.values():
        d["wins"] = sum(bool(f["won"]) for f in d["fights"])
    return starts, teacher, sorted(decks.values(), key=lambda d: (d["wins"], d["source"]))


STARTS_BY_ID, TEACHER_FIGHTS, DECKS = load()
T = tables()
META = dict(cards=T["cards"],
            relics=T["relics"], potions=T["potions"], moves=CHAMP_MOVES, decks=DECKS, pv_model=str(PV_MODEL.relative_to(ROOT)))


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

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


FAST = Session(PV_MODEL if PV_MODEL.exists() else None)
HEAVY = threading.Semaphore(2)


def battle(q):
    """The request's start (bench start + the browser's pre-battle patch) and ops."""
    start = copy.deepcopy(STARTS_BY_ID[q["base"]][1])
    patch = q.get("patch") or {}
    if set(patch) - PATCHABLE:
        raise ValueError(f"cannot patch {set(patch) - PATCHABLE}")
    start.update(patch)
    if "deck" in patch:  # bottled-card indices refer to the original deck
        start["deck"] = [dict(id=int(c["id"]), upgraded=bool(c["upgraded"]), misc=int(c.get("misc", 0))) for c in patch["deck"]]
        start["bottled"] = [-1, -1, -1]
    if "potions" in patch:
        start["potion_capacity"] = len(start["potions"])
    return dict(start=start, ops=q.get("ops", []))


def heavy(request):
    with HEAVY:
        s = Session()
        try:
            return s.ask(request)
        finally:
            s.close()


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


# Finished searches / playouts, keyed by state (base, patch, ops) then by query name, kept across restarts in
# CACHE_FILE: a resumed or pasted link shows them again without recomputing. A failed query is not cached.
CACHE_FILE = ROOT / "experiments/champ-web-viewer/cache.jsonl"
CACHE, CACHE_LOCK = {}, threading.Lock()
if CACHE_FILE.exists():
    for line in CACHE_FILE.open():
        r = json.loads(line)
        CACHE.setdefault(r["state"], {})[r["name"]] = r["result"]


def state_key(base, patch, ops):
    return json.dumps([base, patch or {}, ops or []], sort_keys=True, separators=(",", ":"))


def query_name(q):
    kind, sims = q.get("query", "view"), int(q.get("sims", 20000))
    if kind == "search":
        return f"search sims={sims} salt={int(q.get('salt', 0))}"
    if kind == "search_seeds":
        return f"seeds k={max(1, min(int(q.get('k', 6)), 16))} sims={sims}"
    if kind == "playout":
        return f"playout n={min(int(q.get('n', 10)), 100)} sims={sims}"
    return None


def query(q):
    kind = q.get("query", "view")
    if kind in ("view", "pv"):
        return FAST.ask(dict(battle(q), query=kind))  # query_error (e.g. no PV on a finished fight) keeps the view
    state, name = state_key(q["base"], q.get("patch"), q.get("ops")), query_name(q)
    with CACHE_LOCK:
        hit = CACHE.get(state, {}).get(name)
    if hit is not None:
        return dict(hit, cached=True)
    out = compute(q)
    if name and "error" not in out:
        with CACHE_LOCK:
            CACHE.setdefault(state, {})[name] = out
            with CACHE_FILE.open("a") as f:
                f.write(json.dumps(dict(state=state, name=name, result=out)) + "\n")
    return out


def cached(q):
    """Everything cached for this state, and for each child state reached by one of `children` (action bits)."""
    ops = q.get("ops") or []
    with CACHE_LOCK:
        own = CACHE.get(state_key(q["base"], q.get("patch"), ops), {})
        kids = {str(b): CACHE[k] for b in q.get("children", [])
                if (k := state_key(q["base"], q.get("patch"), ops + [{"act": b}])) in CACHE}
    return dict(own=own, children=kids)


def compute(q):
    req = battle(q)
    kind = q.get("query", "view")
    sims = int(q.get("sims", 20000))
    if kind == "search":
        out = heavy(dict(req, query="search", sims=sims, salt=int(q.get("salt", 0))))
        return dict(error=out["query_error"]) if "query_error" in out else out
    if kind == "search_seeds":  # the same search with k independent seeds: is the visit split stable?
        k = max(1, min(int(q.get("k", 6)), 16))
        with ThreadPoolExecutor(2) as pool:
            outs = list(pool.map(lambda salt: heavy(dict(req, query="search", sims=sims, salt=salt)), range(1, k + 1)))
        for o in outs:
            if "error" in o or "query_error" in o:
                return dict(error=o.get("error") or o["query_error"])
        return dict(runs=[o["search"] for o in outs])
    if kind == "playout":
        n, workers = min(int(q.get("n", 10)), 100), max(1, min(int(q.get("workers", 2)), 2))
        chunks = [(i * n // workers, (i + 1) * n // workers) for i in range(workers)]
        begin = time.time()
        with ThreadPoolExecutor(workers) as pool:
            parts = list(pool.map(lambda c: heavy(dict(req, query="playout", sims=sims, **{"from": c[0], "n": c[1] - c[0]})),
                                  [c for c in chunks if c[1] > c[0]]))
        for p in parts:
            if "error" in p or "query_error" in p:
                return dict(error=p.get("error") or p["query_error"])
        games = [g for p in parts for g in p["playouts"]]
        wins = sum(g["won"] for g in games)
        won_hp = [g["hp"] for g in games if g["won"]]
        return dict(playouts=games, wins=wins, n=len(games), p_win=wins / len(games), ci95=wilson(wins, len(games)),
                    mean_hp_if_won=sum(won_hp) / len(won_hp) if won_hp else None, sims=sims,
                    wall_seconds=time.time() - begin)
    raise ValueError(f"unknown query {kind}")


def teacher_fight(fight_id):
    """The teacher's recorded fight on a bench start: actions and per-decision search rows."""
    f = TEACHER_FIGHTS[fight_id]
    rows = duckdb.sql(f"""select step, root_value, simulations, children from '{TEACHER}/search-*.parquet'
                          where fight_id = '{fight_id}' order by step""").fetchall()
    search = {s: dict(root_value=v, simulations=n,
                      moves={str(c["action"]): dict(visits=c["visits"], value=c["value"]) for c in ch})
              for s, v, n, ch in rows}
    return dict(fight_id=fight_id, won=f["won"], final_hp=f["final_hp"], actions=f["actions"], search=search)


def record(q):
    """Append a finished user fight (start reference, ops incl. edits, outcome) to user_fights.jsonl."""
    row = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), base=q["base"], patch=q.get("patch") or {}, ops=q["ops"],
               undos=q.get("undos", 0), won=q["won"], final_hp=q["final_hp"], note=q.get("note", ""))
    view = FAST.ask(dict(battle(q), query="view"))["view"]  # verify the claimed outcome by rebuilding the fight
    if view["kind"] != "done" or view["won"] != row["won"] or view["player"]["hp"] != row["final_hp"]:
        raise ValueError("recorded outcome does not replay")
    row["edited"] = bool(row["patch"]) or any("act" not in op for op in row["ops"])
    t = TEACHER_FIGHTS.get(q["base"])
    row["teacher_won"] = t and t["won"]
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with RECORD.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return dict(ok=True, edited=row["edited"])


def tally():
    """User vs teacher on the same unedited bench starts (each start counted once: the user's latest record)."""
    latest = {}
    if RECORD.exists():
        for line in RECORD.open():
            r = json.loads(line)
            if not r["edited"]:
                latest[r["base"]] = r
    rows = list(latest.values())
    return dict(n=len(rows), user_wins=sum(r["won"] for r in rows), teacher_wins=sum(bool(r["teacher_won"]) for r in rows),
                user_only=sum(r["won"] and not r["teacher_won"] for r in rows),
                teacher_only=sum(bool(r["teacher_won"]) and not r["won"] for r in rows),
                with_undo=sum(r["undos"] > 0 for r in rows))


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
        if self.path == "/api/champ/meta":
            return self._json(META)
        if self.path == "/api/champ/tally":
            return self._json(tally())
        if self.path.startswith("/api/champ/teacher/"):
            return self._json(teacher_fight(self.path.rsplit("/", 1)[1]))
        return super().do_GET()

    def do_POST(self):
        routes = {"/api/champ/query": query, "/api/champ/record": record, "/api/champ/cached": cached}
        if self.path not in routes:
            return self.send_error(404)
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            self._json(routes[self.path](q))
        except Exception as e:  # a bad edit must not kill the server
            self._json(dict(error=f"{type(e).__name__}: {e}"), 400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8733)
    ap.add_argument("--workers", type=int, default=2, help="max concurrent teacher search / playout processes")
    a = ap.parse_args()
    HEAVY = threading.Semaphore(a.workers)
    print(f"champ viewer: http://127.0.0.1:{a.port}/champ/  ({len(DECKS)} decks, PV model {'on' if FAST else 'off'})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
