"""Static export invariants; no actual runtime download or trained checkpoint needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

from gui.export_gumbel import export, ORT_VERSION


class FrozenExportTest(unittest.TestCase):
    def test_freeze_preserves_rng_and_copies_external_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "date=2026-01-01/id=test/out/training/iter003"
            checkpoint.mkdir(parents=True)
            tensor = helper.make_tensor("weight", TensorProto.FLOAT, [1], bytes(4), raw=True)
            graph = helper.make_graph([], "fixture", [], [], [tensor])
            model = helper.make_model(graph)
            onnx.save_model(model, checkpoint / "model.onnx", save_as_external_data=True,
                            all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=0)
            start = {"seed": 18446744073709551615, "misc_rng": {"seed0": 9007199254740993}}
            fights = root / "fights.jsonl"
            fights.write_text(json.dumps(dict(fight_id="example", start=start)) + "\n")
            wasm, runtime, site = (root / name for name in ("wasm", "runtime", "site"))
            wasm.mkdir()
            (runtime / "dist").mkdir(parents=True)
            for name in ("gumbel.js", "gumbel.wasm"):
                (wasm / name).write_bytes(b"fixture")
            for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
                (runtime / "dist" / name).write_bytes(b"runtime")
            (runtime / "package.json").write_text(json.dumps({"version": ORT_VERSION}))
            export(checkpoint, fights, wasm, runtime, site, 1)
            data = site / "assets/gumbel"
            manifest = json.loads((data / "manifest.json").read_text())
            self.assertEqual(json.loads(manifest["fights"][0]["start_json"]), start)
            self.assertEqual(manifest["external"], ["model.onnx.data"])
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((data / name).read_bytes()).hexdigest(), digest)
            self.assertIn("window.GUMBEL_BROWSER = true", (site / "gumbel-tree/index.html").read_text())
            self.assertTrue((site / ".nojekyll").exists())
            # Re-exporting must not accumulate a second script injection.
            export(checkpoint, fights, wasm, runtime, site, 1)
            self.assertEqual((site / "gumbel-tree/index.html").read_text().count("window.GUMBEL_BROWSER = true"), 1)
            # The frozen model must not change when the training source changes afterwards.
            frozen = (data / "model.onnx.data").read_bytes()
            (checkpoint / "model.onnx.data").write_bytes(b"changed")
            self.assertEqual((data / "model.onnx.data").read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
