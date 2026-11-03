"""Rates from run_actors Counters, as JSON lines written by monitoring/telemetry.hpp."""
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
    requests_s: float      # requests leaving in a batch per second
    wait_ms: float | None  # mean submit-to-batch time; None when nothing left


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
    return [QueueState(id=i, requests_s=d['requests'] / dt,
                       wait_ms=1000 * d['waited_seconds'] / d['requests'] if d['requests'] else None)
            for i, (_, d) in enumerate(rows)]


def lanes(telemetry: Path) -> list[LaneState]:
    rows, dt = deltas(telemetry, 'lanes')
    return [LaneState(id=i, queue=latest['queue'], requests_s=d['requests'] / dt, busy_pct=100 * d['busy_seconds'] / dt,
                      fill_pct=100 * d['requests'] / d['capacity'] if d['capacity'] else None)
            for i, (latest, d) in enumerate(rows)]


def deltas(telemetry: Path, section: str) -> tuple[list[tuple[dict, dict]], float]:
    """Each entry's latest values and increase over the last WINDOW seconds (less, early on), and those seconds."""
    records = []
    for line in telemetry.read_text().splitlines() if telemetry.exists() else []:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a write in progress
    if len(records) < 2:
        return [], 0
    new = records[-1]
    old = next((r for r in reversed(records) if r['time'] <= new['time'] - WINDOW), records[0])
    rows = [(a, {k: v - b[k] for k, v in a.items()}) for a, b in zip(new[section], old[section])]
    return rows, new['time'] - old['time']
