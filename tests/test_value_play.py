"""Orchestration checks for apps/value_play/play.py (no fights played)."""
import contextlib
import hashlib
import json
import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from sts_combat_rl.run import NAME

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("value_play", ROOT / "apps/value_play/play.py")
play = importlib.util.module_from_spec(spec)
spec.loader.exec_module(play)

VALIDATION = [707, 9305, 1203]


@contextlib.contextmanager
def fake_runs(runs):
    """runs: run_id -> inputs; each gets a temp dir with run.json, served by sts_combat_rl.run.run_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {}
        for i, (run_id, inputs) in enumerate(runs.items()):
            dirs[run_id] = Path(tmp) / str(i)
            dirs[run_id].mkdir()
            (dirs[run_id] / "run.json").write_text(json.dumps({"schema": run_id.split("/")[0], "inputs": inputs}))
        with mock.patch("sts_combat_rl.run.run_dir", dirs.__getitem__):
            yield


def config(name):
    return tomllib.loads((ROOT / "apps/value_play" / name).read_text())["run"]


class RunTests(unittest.TestCase):
    def test_bootstrap_inputs_of_dagger_value_run(self):
        runs = {"value_net_v1/d/gen1": ["value_net_v1/d/gen0", "combat_v3/d/boot", "combat_v3/d/dagger"],
                "value_net_v1/d/gen0": ["combat_v3/d/boot"], "combat_v3/d/boot": [], "combat_v3/d/dagger": ["value_net_v1/d/gen0"]}
        with fake_runs(runs):
            self.assertEqual(play.runs({"run": {"input": "value_net_v1/d/gen1"}}),
                             ("value_net_v1/d/gen1", ["combat_v3/d/boot"]))


class EpisodeTests(unittest.TestCase):
    def test_default_is_all_validation(self):
        self.assertEqual(play.select_episodes({}, VALIDATION), VALIDATION)

    def test_subset(self):
        self.assertEqual(play.select_episodes({"episodes": [9305]}, VALIDATION), [9305])

    def test_rejected(self):
        for episodes in ([1], [707, 1], [707, 707], [], ["707"]):
            with self.subTest(episodes=episodes), self.assertRaises(SystemExit):
                play.select_episodes({"episodes": episodes}, VALIDATION)


class TeacherTests(unittest.TestCase):
    def test_legacy_default(self):
        # The worker's default without a teacher object: value_net, random move on.
        self.assertEqual(play.teacher_config({"id": "x", "input": "y", "workers": 2}),
                         {"leaf": "value_net", "random_move": True})
        self.assertEqual(play.teacher_config(config("slime.toml")), {"leaf": "value_net", "random_move": True})

    def test_unknown_key(self):
        with self.assertRaises(SystemExit):
            play.teacher_config({"leaf": "hybrid", "random_moves": False})

    def test_pilot_configs(self):
        expected = {
            "slime4-R.toml": {"leaf": "guided_rollout", "random_move": False},
            "slime4-N.toml": {"leaf": "value_net", "random_move": False},
            "slime4-H.toml": {"leaf": "hybrid", "random_move": False, "rollout_turns": 1, "rollout_steps": 16},
        }
        for name, teacher in expected.items():
            with self.subTest(name=name):
                run = config(name)
                self.assertEqual(play.teacher_config(run), teacher)
                self.assertEqual(run["input"], "value_net_v1/2026-09-23/slime-value-4")
                self.assertEqual(run["workers"], 1)
                self.assertTrue(NAME.match(run["id"]), run["id"])  # sts_combat_rl.run id validation
                self.assertNotIn("episodes", run)  # all validation episodes


class WorkerTests(unittest.TestCase):
    def test_argv(self):
        self.assertEqual(play.worker_argv("/r/bin", None, "f.json", "t"), ["/r/bin", "--no-weights", "f.json", "t"])
        self.assertEqual(play.worker_argv("/r/bin", Path("w.bin"), "f.json", "t"), ["/r/bin", "w.bin", "f.json", "t"])

    def test_play_runs_given_binary(self):
        calls = []

        def run(argv, check):
            calls.append(argv)
            raise RuntimeError("stop")

        with tempfile.TemporaryDirectory() as out, mock.patch.object(play.subprocess, "run", run):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                play.play(707, {"teacher": {}}, {}, Path(out) / "value_play_worker", None, Path(out))
        self.assertEqual(calls[0][0], str(Path(out) / "value_play_worker"))
        self.assertEqual(calls[0][1], "--no-weights")

    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            built, out = Path(tmp) / "built", Path(tmp) / "out"
            out.mkdir()
            built.write_bytes(b"worker v1")
            with mock.patch.object(play, "BUILT", built):
                binary, sha = play.snapshot_worker(out)
            built.write_bytes(b"worker v2")  # a later rebuild doesn't change the run's copy
            self.assertEqual(binary, (out / "value_play_worker").resolve())
            self.assertEqual(binary.read_bytes(), b"worker v1")
            self.assertEqual(sha, hashlib.sha256(b"worker v1").hexdigest())
            self.assertFalse(binary.stat().st_mode & 0o222)  # read-only


if __name__ == "__main__":
    unittest.main()
