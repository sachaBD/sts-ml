"""Small terminal progress display for value_play."""
from __future__ import annotations

import os
import sys
import time


class Progress:
    """Redraw a compact aggregate table when launched through a live terminal."""
    def __init__(self, total: int, leaf: str):
        self.total, self.leaf, self.done = total, leaf, []
        self.live = os.environ.get("RUN_INTERACTIVE") == "1"
        self.lines = 8
        self.drawn = False
        self.last_draw = 0.0
        self.started = time.monotonic()
        self.rendered = 0

    def add(self, fight: dict) -> bool:
        self.done.append(fight)
        now = time.monotonic()
        if self.live and (len(self.done) == self.total or now - self.last_draw >= 0.2):
            self.draw()
        return not self.live and (len(self.done) % 10 == 0 or len(self.done) == self.total)

    def draw(self) -> None:
        n = len(self.done)
        mean = lambda key: sum(f[key] for f in self.done) / n if n else 0.0
        teacher_wins = sum(f["teacher"]["won"] for f in self.done)
        decisions = sum(f["decisions"] for f in self.done)
        simulations = sum(f["simulations"] for f in self.done)
        eta = ((time.monotonic() - self.started) / n * (self.total - n)) if n else 0.0
        bar = "#" * (20 * n // self.total) + "." * (20 - 20 * n // self.total)
        replay_wins = sum(f["won"] for f in self.done)
        replay_value, stored_value = mean("terminal_value"), mean_teacher(self.done)
        rows = [
            f"Value play [{bar}] {n}/{self.total}  ETA {eta / 60:.1f} min",
            f"{'':24}{'replay':>12}{'stored':>12}{'delta':>10}",
            f"{'wins':24}{f'{replay_wins}/{n}':>12}{f'{teacher_wins}/{n}':>12}{replay_wins - teacher_wins:>+10}",
            f"{'mean terminal value':24}{replay_value:>12.3f}{stored_value:>12.3f}{replay_value - stored_value:>+10.3f}",
            f"{'mean final hp':24}{mean('final_hp'):>12.1f}{mean_teacher_hp(self.done):>12.1f}{'':>10}",
            "",
            f"performance: {simulations / decisions if decisions else 0:,.0f} sims/decision  "
            f"{mean('seconds'):.1f}s/fight  {decisions / n if n else 0:.1f} decisions/fight",
            f"last: episode {self.done[-1]['episode'] if n else '-'}  ({self.leaf})",
        ]
        prefix = f"\x1b[{self.lines}A" if self.drawn else ""
        sys.stdout.write(prefix + "".join(f"\x1b[2K{row}\n" for row in rows))
        sys.stdout.flush()
        self.drawn, self.last_draw, self.rendered = True, time.monotonic(), n

    def finish(self) -> None:
        if self.live and self.done and self.rendered != len(self.done):
            self.draw()


def mean_teacher(fights: list[dict]) -> float:
    return sum(f["teacher"]["terminal_value"] for f in fights) / len(fights) if fights else 0.0


def mean_teacher_hp(fights: list[dict]) -> float:
    return sum(f["teacher"]["final_hp"] for f in fights) / len(fights) if fights else 0.0
