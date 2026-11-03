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


def actors(telemetry: Path) -> list[ActorState]:
    rows, dt = deltas(telemetry, 'actors')
    return [ActorState(id=i, requests_s=d['requests'] / dt, blocked_pct=100 * d['blocked_seconds'] / dt)
            for i, d in enumerate(rows)]


def deltas(telemetry: Path, section: str) -> tuple[list[dict], float]:
    """Each entry's counter increase over the last WINDOW seconds (less, early on), and those seconds."""
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
    rows = [{k: v - b[k] for k, v in a.items()} for a, b in zip(new[section], old[section])]
    return rows, new['time'] - old['time']
