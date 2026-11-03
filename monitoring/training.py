"""TrainingCounters JSON lines and recent optimizer-step rates/losses."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .telemetry import WINDOW


def write(stream, counters):
    stream.write(json.dumps(asdict(counters)) + '\n')
    stream.flush()


@dataclass(frozen=True)
class TrainingState:
    step: int
    total: int
    seconds: float         # elapsed training wall time, not scoring/loading/export
    window_seconds: float
    steps_s: float
    value_loss: float      # mean per optimizer step over the counter window
    policy_loss: float


def training(path: Path) -> TrainingState | None:
    records = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a write in progress
    if len(records) < 2:
        return None
    new = records[-1]
    old = next((r for r in reversed(records) if r['seconds'] <= new['seconds'] - WINDOW), records[0])
    seconds, steps = new['seconds'] - old['seconds'], new['step'] - old['step']
    if seconds <= 0 or steps <= 0:
        return None
    return TrainingState(step=new['step'], total=new['total'], seconds=new['seconds'], window_seconds=seconds,
                         steps_s=steps / seconds,
                         value_loss=(new['value_loss_sum'] - old['value_loss_sum']) / steps,
                         policy_loss=(new['policy_loss_sum'] - old['policy_loss_sum']) / steps)
