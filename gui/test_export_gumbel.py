import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

from gui.export_gumbel import export, ORT_VERSION


class FrozenExportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.checkpoint = root / "date=2026-01-01/id=test/out/training/iter003"
        self.checkpoint.mkdir(parents=True)
        tensor = helper.make_tensor("weight", TensorProto.FLOAT, [1], bytes(4), raw=True)
        model = helper.make_model(helper.make_graph([], "fixture", [], [], [tensor]))
        onnx.save_model(model, self.checkpoint / "model.onnx", save_as_external_data=True,
                        all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=0)
        self.rows = [dict(fight_id=str(i), start={"seed": seed, "misc_rng": {"seed0": seed}})
                     for i, seed in enumerate((0, 2**53 - 1, 2**53, 2**53 + 1, 2**63, 2**64 - 1))]
        self.fights = root / "fights.jsonl"
        self.fights.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.wasm, self.runtime, self.site = (root / name for name in ("wasm", "runtime", "site"))
        self.wasm.mkdir()
        (self.runtime / "dist").mkdir(parents=True)
        for name in ("gumbel.js", "gumbel.wasm"):
            (self.wasm / name).write_bytes(name.encode())
        for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
            (self.runtime / "dist" / name).write_bytes(name.encode())
        (self.runtime / "package.json").write_text(json.dumps({"version": ORT_VERSION}))

    def freeze(self, count=None):
        export(self.checkpoint, self.fights, self.wasm, self.runtime, self.site,
               len(self.rows) if count is None else count)
        return json.loads((self.site / "assets/gumbel/manifest.json").read_text())

    def snapshot(self):
        return {path.relative_to(self.site): path.read_bytes() for path in self.site.rglob("*") if path.is_file()}

    def test_fight_starts_survive_json_round_trip_without_numeric_loss(self):
        manifest = self.freeze()
        original = {row["fight_id"]: row["start"] for row in self.rows}
        for fight in manifest["fights"]:
            with self.subTest(fight=fight["id"]):
                self.assertIsInstance(fight["start_json"], str)
                self.assertEqual(json.loads(fight["start_json"]), original[fight["id"]])

    def test_manifest_digests_match_published_bytes(self):
        manifest = self.freeze()
        sources = [*self.checkpoint.iterdir(), *self.wasm.iterdir(), *(self.runtime / "dist").iterdir()]
        self.assertEqual(set(manifest["sha256"]), {source.name for source in sources})
        for source in sources:
            with self.subTest(artifact=source.name):
                payload = (self.site / "assets/gumbel" / source.name).read_bytes()
                self.assertEqual(payload, source.read_bytes())
                self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["sha256"][source.name])

    def test_published_model_is_independent_of_source_checkpoint(self):
        self.freeze()
        before = self.snapshot()
        for source in self.checkpoint.iterdir():
            source.write_bytes(b"replaced")
            source.unlink()
        self.assertEqual(self.snapshot(), before)

    def test_export_is_idempotent_for_unchanged_inputs(self):
        self.freeze()
        before = self.snapshot()
        self.freeze()
        self.assertEqual(self.snapshot(), before)

    def test_selection_is_a_bounded_unique_subset_of_inputs(self):
        original = {row["fight_id"]: row["start"] for row in self.rows}
        for count in range(1, len(self.rows) + 3):
            with self.subTest(count=count):
                selected = self.freeze(count)["fights"]
                ids = [fight["id"] for fight in selected]
                self.assertEqual(len(ids), min(count, len(self.rows)))
                self.assertEqual(len(set(ids)), len(ids))
                for fight in selected:
                    self.assertIn(fight["id"], original)
                    self.assertEqual(json.loads(fight["start_json"]), original[fight["id"]])


if __name__ == "__main__":
    unittest.main()
