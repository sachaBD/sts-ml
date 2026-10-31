"""Synthetic-only tests: no native processes, GPU access or combat runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark
from report import render, summarize


class BenchmarkTest(unittest.TestCase):
    def test_command_preserves_historical_contract(self):
        config = dict(search=dict(sims=256, particles=8, gumbel_scale=0.0), workers=dict(
            actors=9, slots=224, gpu_batch=512, gpu_bank=[256, 32], gpu_wait_us=100,
            timeout_ms=60000, shutdown_timeout_ms=5000))
        paths = {n: Path('/frozen') / n for n in ('worker', 'model.onnx', 'model-gpu.onnx',
                                                  'model-gpu-b256.onnx', 'model-gpu-b32.onnx')}
        a = benchmark.command(config, paths, 0.25, Path('/output'))
        b = benchmark.command(config, paths, 0.0, Path('/output'))
        self.assertEqual(a[:4], ['/frozen/worker', 'play', '/frozen/model.onnx', '/output/samples-0.bin'])
        flags = dict(zip(a[4::2], a[5::2]))
        self.assertEqual(flags['--gpu-bank'], '/frozen/model-gpu-b256.onnx,/frozen/model-gpu-b32.onnx')
        self.assertEqual(flags['--sims'], '256')
        self.assertEqual(flags['--particles'], '8')
        self.assertEqual(flags['--gumbel-scale'], '0.0')
        self.assertEqual(a[:-1], b[:-1])
        self.assertEqual(a[-1], '0.25')
        self.assertEqual(b[-1], '0.0')

    def test_invalid_results_rejected(self):
        row = dict(fight_id='ns:deck:monitor:0', won=True)
        for rows in ([row, row], [dict(row, capped=True)], [dict(row, won=1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                benchmark.outcomes(rows)

    def test_paired_report(self):
        keys = [f'ns:deck:monitor:{i}' for i in range(4)]
        old = dict(zip(keys, [False, True, False, True]))
        new = dict(zip(keys, [True, True, False, True]))
        summary = summarize(new, dict(nn50=old))
        self.assertEqual(summary['aggregate']['wins'], 3)
        self.assertEqual(summary['aggregate']['comparisons']['nn50']['gap'], 0.25)
        self.assertEqual(summary['decks']['deck']['vs_nn50']['d'], 0.25)
        self.assertIn('historical / not comparable', render(dict(nn50=summarize(old, dict(nn50=old)))))
        with self.assertRaises(ValueError):
            summarize(dict(list(new.items())[:-1]), dict(nn50=old))

    def test_preflight_checks_hashes_and_reference_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fights = [dict(fight_id=f'ns:deck{d}:monitor:{i}', start=dict(seed=i))
                      for d in range(10) for i in range(50)]
            rows = [dict(fight_id=f['fight_id'], won=False) for f in fights]
            files = dict(fights='\n'.join(map(json.dumps, fights)), nn50='\n'.join(map(json.dumps, rows)),
                         historical=json.dumps(dict(mcts20k={r['fight_id']:False for r in rows})))
            artifacts = {}
            for name, contents in files.items():
                path = directory / name
                path.write_text(contents)
                artifacts[name] = dict(path=str(path), sha256=benchmark.digest(path))
            config = dict(artifacts=artifacts)
            with patch('benchmark.subprocess.run', side_effect=AssertionError('preflight launched process')):
                self.assertEqual(len(benchmark.verify(config)[1]), 500)
            (directory / 'nn50').write_text(files['nn50'] + '\n' + json.dumps(rows[0]))
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                benchmark.verify(config)
            artifacts['nn50']['sha256'] = benchmark.digest(directory / 'nn50')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                benchmark.verify(config)


if __name__ == '__main__':
    unittest.main()
