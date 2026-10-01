"""RETIRED (2026-10-01): the bandit approach to Neow is abandoned. The Neow moment is NOT a fixed state: the map
and the act boss are already known and differ per run, so a context-free per-arm table is the wrong model. Neow is
to be treated as a regular decision scored by the value network (see docs/research/neow_handoff.md). Kept only so the
existing play.py / loop.py wiring still runs; do not build on it.

Neow choice policy for the real-run RL loop: a bandit over Neow ARMS (docs/research/run-rl/neow.md).

An arm = "<bonus>|<drawback>" (worker.cpp neow_bonus_names / neow_drawback_names), e.g. "hundred_gold|none",
"boss_relic|lose_starter_relic". Each run offers 4 arms; the policy picks one (option index 0-3).

INTERFACE (used by play.py / loop.py; keep it):
    p = NeowPolicy.load(path, seed=0)           # path None or missing file -> fresh policy
    i = p.choose(options, explore)              # options: worker labels [{"index", "bonus", "drawback", "arm"}] x 4
                                                # explore True (training batches) / False (eval: greedy)
    p.update(runs, score_fn)                    # runs: finished logged runs (play.py runs.jsonl lines);
                                                # each has one step {"kind": "decide", "decision": "neow",
                                                # "options", "choice"}; score_fn(run) -> reward in [0, 1]
    p.save(path)                                # JSON, human-readable
    p.summary() -> list of dicts                # per arm: arm, n, mean, ... (for reports)

Thompson sampling: per arm a Beta(1 + s, 1 + f) posterior, s = discounted sum of scores, f = discounted sum of
(1 - score). Each update() first multiplies all existing evidence by `discount` (the rest of the policy keeps
changing, so old runs count less), then adds the batch. explore=True draws one sample per offered arm and takes the
highest; explore=False takes the highest posterior mean (ties -> lowest index, i.e. SimpleAgent's slot-1 choice).
"""
import json
import random
from pathlib import Path


class NeowPolicy:
    def __init__(self, arms=None, discount=0.9, seed=0):
        self.arms = arms or {}  # arm -> {"s": success weight, "f": failure weight, "n": raw count (undiscounted)}
        self.discount = discount
        self.rng = random.Random(seed)

    @classmethod
    def load(cls, path, seed=0, discount=None):
        if path and Path(path).exists():
            d = json.loads(Path(path).read_text())
            return cls(d["arms"], discount if discount is not None else d.get("discount", 0.9), seed)
        return cls(discount=0.9 if discount is None else discount, seed=seed)

    def save(self, path):
        Path(path).write_text(json.dumps({"discount": self.discount, "arms": self.arms, "summary": self.summary()},
                                         indent=1, sort_keys=True))

    def _ab(self, arm):
        a = self.arms.get(arm, {"s": 0.0, "f": 0.0})
        return 1.0 + a["s"], 1.0 + a["f"]

    def choose(self, options, explore):
        def value(o):
            a, b = self._ab(o["arm"])
            return self.rng.betavariate(a, b) if explore else a / (a + b)
        vals = [value(o) for o in options]
        return max(range(len(options)), key=lambda i: (vals[i], -i))

    def update(self, runs, score_fn):
        for a in self.arms.values():
            a["s"] *= self.discount
            a["f"] *= self.discount
        for run in runs:
            for s in run["steps"]:
                if s["kind"] == "decide" and s["decision"] == "neow":
                    arm = s["options"][s["choice"]]["arm"]
                    a = self.arms.setdefault(arm, {"s": 0.0, "f": 0.0, "n": 0})
                    r = min(max(float(score_fn(run)), 0.0), 1.0)
                    a["s"] += r
                    a["f"] += 1.0 - r
                    a["n"] += 1

    def summary(self):
        rows = []
        for arm, v in self.arms.items():
            a, b = self._ab(arm)
            sd = (a * b / ((a + b) ** 2 * (a + b + 1))) ** 0.5
            rows.append({"arm": arm, "n": v["n"], "weight": round(v["s"] + v["f"], 1),
                         "mean": round(a / (a + b), 4), "sd": round(sd, 4)})
        return sorted(rows, key=lambda r: -r["mean"])
