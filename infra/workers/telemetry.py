"""Rates from run_actors Counters, as JSON lines written by infra/workers/telemetry.hpp."""
from dataclasses import dataclass
import json
from pathlib import Path

WINDOW = 5  # seconds of counters behind each rate


@dataclass(frozen=True)
class ActorState:
    id: int
    requests_s: float   # inference requests submitted per second
    blocked_pct: float  # percent of wall time waiting on inference replies


@dataclass(frozen=True)
class QueueState:
    id: int                # backend index
    backend: str           # the backend's name
    requests: int          # requests that left in a batch since run_actors began
    share_pct: float       # percent of every queue's requests since run_actors began
    requests_s: float      # requests leaving in a batch per second
    wait_ms: float | None  # mean submit-to-batch time; None when nothing left
    depth: int             # pending count at latest sample
    peak_depth: int        # highest pending count since run_actors began


@dataclass(frozen=True)
class LaneState:
    id: int
    queue: int              # the QueueState this lane serves
    requests_s: float       # requests evaluated per second
    busy_pct: float         # percent of wall time evaluating and replying
    fill_pct: float | None  # requests / batch capacity; None when no batch ran


def actors(telemetry: Path) -> list[ActorState]:
    rows, dt = deltas(telemetry, 'actors')
    return [ActorState(id=i, requests_s=d['requests'] / dt, blocked_pct=100 * d['blocked_seconds'] / dt)
            for i, (_, d) in enumerate(rows)]


def queues(telemetry: Path) -> list[QueueState]:
    rows, dt = deltas(telemetry, 'queues')
    total = sum(latest['requests'] for latest, _ in rows)
    return [QueueState(id=i, backend=latest['backend'], requests=latest['requests'],
                       share_pct=100 * latest['requests'] / total if total else 0, requests_s=d['requests'] / dt,
                       wait_ms=1000 * d['waited_seconds'] / d['requests'] if d['requests'] else None,
                       depth=latest['depth'], peak_depth=latest['peak_depth'])
            for i, (latest, d) in enumerate(rows)]


def lanes(telemetry: Path) -> list[LaneState]:
    rows, dt = deltas(telemetry, 'lanes')
    return [LaneState(id=i, queue=latest['queue'], requests_s=d['requests'] / dt, busy_pct=100 * d['busy_seconds'] / dt,
                      fill_pct=100 * d['requests'] / d['capacity'] if d['capacity'] else None)
            for i, (latest, d) in enumerate(rows)]


def evaluated_per_second(telemetry: Path) -> float | None:
    """Requests every lane evaluated per second, over the whole of run_actors; None before two records."""
    rows = read(telemetry)
    if len(rows) < 2 or rows[-1]['time'] <= rows[0]['time']:
        return None
    evaluated = [sum(lane['requests'] for lane in r['lanes']) for r in (rows[0], rows[-1])]
    return (evaluated[1] - evaluated[0]) / (rows[-1]['time'] - rows[0]['time'])


def read(telemetry: Path) -> list[dict]:
    rows = []
    for line in telemetry.read_text().splitlines() if telemetry.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a write in progress
    return rows


def deltas(telemetry: Path, section: str) -> tuple[list[tuple[dict, dict]], float]:
    """Each entry's latest values and increase over the last WINDOW seconds (less, early on), and those seconds."""
    records = read(telemetry)
    if len(records) < 2:
        return [], 0
    new = records[-1]
    old = next((r for r in reversed(records) if r['time'] <= new['time'] - WINDOW), records[0])
    rows = [(a, {k: v - b[k] for k, v in a.items() if k != 'backend'}) for a, b in zip(new[section], old[section])]
    return rows, new['time'] - old['time']
