"""Neow choice policy for the real-run RL loop: a bandit over Neow ARMS (slop_docs/run-rl/neow.md).

An arm = "<bonus>|<drawback>" (worker.cpp neow_bonus_names / neow_drawback_names), e.g. "hundred_gold|none",
"boss_relic|lose_starter_relic". Each run offers 4 arms; the policy picks one (option index 0-3).

INTERFACE (used by play.py / loop.py; keep it):
    p = NeowPolicy.load(path)                   # path None or missing file -> fresh policy
    i = p.choose(options, explore)              # options: worker labels [{"index", "bonus", "drawback", "arm"}] x 4
                                                # explore True (training batches) / False (eval: greedy)
    p.update(runs, score_fn)                    # runs: finished logged runs (play.py runs.jsonl lines);
                                                # each has one step {"kind": "decide", "decision": "neow",
                                                # "options", "choice"}; score_fn(run) -> reward in [0, 1]
    p.save(path)                                # JSON, human-readable
    p.summary() -> list of dicts                # per arm: arm, n, mean, ... (for reports)

STATUS: stub. choose() always returns 0 (= SimpleAgent's Neow choice), update() only counts. The bandit
(Thompson sampling, discounting) is to be implemented per slop_docs/run-rl/neow.md "Implementation spec".
"""
import json
from pathlib import Path


class NeowPolicy:
    def __init__(self, arms=None):
        self.arms = arms or {}  # arm -> {"n": count, "sum": reward sum}  (the implementer may change the fields)

    @classmethod
    def load(cls, path):
        if path and Path(path).exists():
            return cls(json.loads(Path(path).read_text())["arms"])
        return cls()

    def save(self, path):
        Path(path).write_text(json.dumps({"arms": self.arms}, indent=1, sort_keys=True))

    def choose(self, options, explore):
        return 0  # STUB: SimpleAgent's choice

    def update(self, runs, score_fn):
        for run in runs:
            for s in run["steps"]:
                if s["kind"] == "decide" and s["decision"] == "neow":
                    arm = s["options"][s["choice"]]["arm"]
                    a = self.arms.setdefault(arm, {"n": 0, "sum": 0.0})
                    a["n"] += 1
                    a["sum"] += score_fn(run)

    def summary(self):
        return sorted(({"arm": k, "n": v["n"], "mean": v["sum"] / max(v["n"], 1)} for k, v in self.arms.items()),
                      key=lambda r: -r["mean"])
