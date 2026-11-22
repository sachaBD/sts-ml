import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

from apps.combat_inspector.pages import export, ORT_VERSION, ROOT


class PagesTest(unittest.TestCase):
    def test_frozen_export_preserves_rng_bits_and_hashes_and_replaces_only_its_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, wasm, runtime, site = [root / name for name in ('checkpoint', 'wasm', 'runtime', 'site')]
            checkpoint.mkdir()
            (checkpoint / 'network.toml').write_text('encoding="inventory_v1"\n[cpu]\nfile="model.onnx"\n')
            tensor = helper.make_tensor('weight', TensorProto.FLOAT, [1], bytes(4), raw=True)
            model = helper.make_model(helper.make_graph([], 'fixture', [], [], [tensor]))
            onnx.save_model(model, checkpoint / 'model.onnx', save_as_external_data=True,
                            all_tensors_to_one_file=True, location='weights.data', size_threshold=0)
            wasm.mkdir()
            (runtime / 'dist').mkdir(parents=True)
            (runtime / 'package.json').write_text(json.dumps(dict(version=ORT_VERSION)))
            for name in ('gumbel.js', 'gumbel.wasm'):
                (wasm / name).write_bytes(name.encode())
            for name in ('ort.wasm.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'):
                (runtime / 'dist' / name).write_bytes(name.encode())
            starts = [dict(fight_id=str(i), start=dict(seed=n)) for i, n in enumerate((2**53 + 1, 2**64 - 1))]
            fights = root / 'fights.jsonl'
            fights.write_text(''.join(json.dumps(row) + '\n' for row in starts))
            manifest = export(checkpoint, fights, wasm, runtime, site, ROOT)
            self.assertEqual([json.loads(f['start_json']) for f in manifest['fights']], [r['start'] for r in starts])
            for name, digest in manifest['sha256'].items():
                self.assertEqual(hashlib.sha256((site / 'assets/gumbel' / name).read_bytes()).hexdigest(), digest)
            unrelated = site / 'another-report.html'
            unrelated.write_text('keep me')
            stale = site / 'gumbel-tree/stale.js'
            stale.touch()
            again = export(checkpoint, fights, wasm, runtime, site, ROOT)
            self.assertEqual(manifest['checkpoint_id'], again['checkpoint_id'])
            self.assertFalse(stale.exists())
            self.assertEqual(unrelated.read_text(), 'keep me')


if __name__ == '__main__':
    unittest.main()
