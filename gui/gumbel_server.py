"""Gumbel tree viewer backend, mounted by gui/server.py under /api/gumbel/.

Lists gumbel_nn_combat_v1 runs, their checkpoints and evaluation fights, and answers one decision with its whole
search tree from nn_explain (agents/combat/nn/explain.cpp). The browser names a fight by (run, fight id); starts stay
here, since their 64-bit seeds do not survive JavaScript numbers.
"""
import json
import subprocess
import threading
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOTS = [ROOT / "runs"]
EXPLAIN = ROOT / "build/main/agents/combat/nn/nn_explain"


def runs(q):
    found = [d for root in RUN_ROOTS for d in sorted(root.glob("schema=gumbel_nn_combat_v1/date=*/id=*"))
             if (d / "out/inputs/test.jsonl").exists()]
    return dict(runs=[dict(run=str(d), date=d.parent.name.removeprefix("date="), id=d.name.removeprefix("id="))
                      for d in found])


def run_dir(path):
    if path not in {r["run"] for r in runs({})["runs"]}:
        raise ValueError(f"not a known run: {path}")
    return Path(path) / "out"


@cache
def starts(fights_file):
    with open(fights_file) as f:
        return {fight["fight_id"]: fight["start"] for fight in map(json.loads, filter(str.strip, f))}


def run(q):
    out = run_dir(q["run"])
    iterations = sorted(int(d.name.removeprefix("iter")) for d in (out / "training").glob("iter*")
                        if (d / "model.onnx").exists())
    return dict(iterations=iterations, fights=list(starts(out / "inputs/test.jsonl")))


class Explainer:
    """One nn_explain process: a request line in, a response line out."""

    def __init__(self):
        self.proc, self.lock = None, threading.Lock()

    def ask(self, request):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.proc = subprocess.Popen([str(EXPLAIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"nn_explain died (exit {self.proc.poll()})")
        return json.loads(line)


EXPLAINER = Explainer()


def decision(q):
    out = run_dir(q["run"])
    model = out / f"training/iter{int(q['iteration']):03d}/model.onnx"
    if not model.exists():
        raise ValueError(f"no model for iteration {q['iteration']}")
    start = starts(out / "inputs/test.jsonl")[q["fight"]]
    return EXPLAINER.ask(dict(model=str(model), start=start, moves=q["moves"], search=q["search"]))
