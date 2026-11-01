"""Gumbel tree viewer server, on a temporary run with an nn_explain stand-in that echoes each request."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gumbel_server  # noqa: E402

START = dict(seed=2**64 - 59, hp=27)  # a seed above 2**53: exact only if it never passes through a float
ECHO = f"#!{sys.executable}\nimport sys\nfor line in sys.stdin:\n    print(line, end='', flush=True)\n"


class GumbelServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.run = root / "runs/schema=gumbel_nn_combat_v1/date=2026-01-01/id=r"
        (self.run / "out/inputs").mkdir(parents=True)
        (self.run / "out/training/iter003").mkdir(parents=True)
        (self.run / "out/inputs/test.jsonl").write_text(json.dumps(dict(fight_id="f", start=START)) + "\n")
        (self.run / "out/training/iter003/model.onnx").touch()
        echo = root / "nn_explain"
        echo.write_text(ECHO)
        echo.chmod(0o755)
        explainer = gumbel_server.Explainer()
        for name, value in dict(RUN_ROOTS=[root / "runs"], EXPLAIN=echo, EXPLAINER=explainer).items():
            patch = mock.patch.object(gumbel_server, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(lambda: explainer.proc and explainer.proc.communicate())

    def test_only_discovered_runs_are_read(self):
        self.assertEqual([r["run"] for r in gumbel_server.runs({})["runs"]], [str(self.run)])
        for outside in (self.tmp.name, str(self.run / ".."), "/etc"):
            with self.assertRaises(ValueError):
                gumbel_server.run(dict(run=outside))

    def test_a_decision_asks_for_the_fight_exactly(self):
        info = gumbel_server.run(dict(run=str(self.run)))
        self.assertEqual((info["iterations"], info["fights"]), ([3], ["f"]))
        search = dict(sims=64, particles=8, gumbel_scale=0, rollout_mix=0.5)
        asked = gumbel_server.decision(dict(run=str(self.run), iteration=3, fight="f", moves=[7], search=search))
        self.assertEqual(asked, dict(model=str(self.run / "out/training/iter003/model.onnx"), start=START, moves=[7],
                                     search=search))


if __name__ == "__main__":
    unittest.main()
