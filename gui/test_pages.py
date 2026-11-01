import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

from gui.export_gumbel import ORT_VERSION
from gui.pages import git, prepare, publish


class PagesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.remote, self.repository = root / "remote.git", root / "checkout"
        self.repository.mkdir()
        git(root, "init", "--bare", str(self.remote))
        git(self.repository, "init", "-b", "gh-pages")
        git(self.repository, "config", "user.email", "pages@example.invalid")
        git(self.repository, "config", "user.name", "Pages test")
        git(self.repository, "remote", "add", "origin", str(self.remote))
        self.preserved = {"other-report/index.html": b"other report", "assets/unrelated.bin": bytes(range(256))}
        files = {**self.preserved, "index.html": b"old landing", ".gitignore": b"build/\n",
                 "assets/gumbel/obsolete.bin": b"old model", "gumbel-tree/obsolete.js": b"old script"}
        for name, content in files.items():
            path = self.repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        git(self.repository, "add", ".")
        git(self.repository, "commit", "-m", "Initial site")
        git(self.repository, "push", "origin", "gh-pages")
        self.site = self.repository / "build/pages/site"
        self.checkpoint = root / "run/out/training/iter001"
        self.checkpoint.mkdir(parents=True)
        tensor = helper.make_tensor("weight", TensorProto.FLOAT, [1], bytes(4), raw=True)
        onnx.save_model(helper.make_model(helper.make_graph([], "fixture", [], [], [tensor])),
                        self.checkpoint / "model.onnx")
        self.fights = root / "fights.jsonl"
        self.fights.write_text(json.dumps({"fight_id": "fixture", "start": {"seed": 2**64 - 1}}) + "\n")
        self.wasm, self.runtime = root / "wasm", root / "runtime"
        self.wasm.mkdir()
        (self.runtime / "dist").mkdir(parents=True)
        for name in ("gumbel.js", "gumbel.wasm"):
            (self.wasm / name).write_bytes(name.encode())
        for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
            (self.runtime / "dist" / name).write_bytes(name.encode())
        (self.runtime / "package.json").write_text(json.dumps({"version": ORT_VERSION}))

    def prepare(self):
        prepare(self.site, self.checkpoint, self.fights, self.wasm, self.runtime, 1, repository=self.repository)

    def test_preparation_cannot_replace_source_directories(self):
        for destination in (self.repository, self.repository / "gui", self.repository / "build"):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "inside build/"):
                    prepare(destination, self.checkpoint, self.fights, self.wasm, self.runtime, 1,
                            repository=self.repository)
        self.assertEqual(git(self.repository, "status", "--porcelain"), "")

    def test_unrelated_reports_survive_prepare_and_publish(self):
        source_head = git(self.repository, "rev-parse", "HEAD")
        self.prepare()
        for name, content in self.preserved.items():
            self.assertEqual((self.site / name).read_bytes(), content)
        publish(self.site, repository=self.repository)
        for name, content in self.preserved.items():
            published = subprocess.check_output(["git", "-C", str(self.remote), "show", f"gh-pages:{name}"])
            self.assertEqual(published, content)
        self.assertEqual(git(self.repository, "rev-parse", "HEAD"), source_head)
        self.assertEqual(git(self.repository, "status", "--porcelain"), "")

    def test_changes_to_prepared_snapshot_prevent_publication(self):
        before = git(self.remote, "rev-parse", "gh-pages")
        for change in ("edit", "delete", "add"):
            with self.subTest(change=change):
                self.prepare()
                path = self.site / "index.html"
                if change == "edit":
                    path.write_bytes(b"modified")
                elif change == "delete":
                    path.unlink()
                else:
                    (self.site / "unexpected").touch()
                with self.assertRaisesRegex(ValueError, "prepared site changed"):
                    publish(self.site, repository=self.repository)
                self.assertEqual(git(self.remote, "rev-parse", "gh-pages"), before)

    def test_concurrent_remote_updates_cannot_be_overwritten(self):
        self.prepare()
        (self.repository / "concurrent.txt").write_text("another release")
        git(self.repository, "add", ".")
        git(self.repository, "commit", "-m", "Concurrent deployment")
        git(self.repository, "push", "origin", "gh-pages")
        before = git(self.remote, "rev-parse", "gh-pages")
        with self.assertRaisesRegex(ValueError, "gh-pages changed"):
            publish(self.site, repository=self.repository)
        self.assertEqual(git(self.remote, "rev-parse", "gh-pages"), before)

    def test_replaced_reports_do_not_retain_obsolete_artifacts(self):
        self.prepare()
        for obsolete in ("assets/gumbel/obsolete.bin", "gumbel-tree/obsolete.js"):
            with self.subTest(path=obsolete):
                self.assertFalse((self.site / obsolete).exists())

    def test_unchanged_release_does_not_create_another_commit(self):
        self.prepare()
        publish(self.site, repository=self.repository)
        before = git(self.remote, "rev-parse", "gh-pages")
        self.prepare()
        publish(self.site, repository=self.repository)
        self.assertEqual(git(self.remote, "rev-parse", "gh-pages"), before)


if __name__ == "__main__":
    unittest.main()
