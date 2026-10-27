"""Opt-in real-model checks. Default CTest runs skip these; no recorded-run discovery.

Set NN_TEST_MODEL and NN_TEST_STARTS. Optional NN_TEST_GPU, NN_TEST_GPU_BANK,
NN_TEST_GPU_BATCH (default512), NN_TEST_OUT preserve native smoke artifacts.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(os.environ.get('NN_TEST_MODEL') and os.environ.get('NN_TEST_STARTS'), 'explicit frozen model/starts required')
class WorkerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(os.environ.get('NN_TEST_OUT', self.temporary.name))
        self.out.mkdir(parents=True, exist_ok=True)
        self.starts = Path(os.environ['NN_TEST_STARTS']).read_text().splitlines()

    def play(self, name, count, options):
        directory = self.out / name
        directory.mkdir(exist_ok=False)
        binary = os.environ.get('NN_TEST_BINARY', str(ROOT / 'build/nn/agents/combat/nn/nn_play'))
        command = [binary, 'play', os.environ['NN_TEST_MODEL'], str(directory / 'samples.bin'),
                   '--rollout-mix', '.5', '--gumbel-scale', '1', *options]
        (directory / 'command.json').write_text(json.dumps(command))
        packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        libraries = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libraries)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        with (directory / 'worker.log').open('w') as errors:
            result = subprocess.run(command, input='\n'.join(self.starts[:count])+'\n', text=True,
                                    stdout=subprocess.PIPE, stderr=errors, env=env, timeout=90, check=True)
        (directory / 'results.jsonl').write_text(result.stdout)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        expected = {json.loads(line)['fight_id'] for line in self.starts[:count]}
        self.assertEqual(len(rows), count)
        self.assertEqual({r['fight_id'] for r in rows}, expected)
        # Validate output format, complete sample records, probability targets, and sample counts.
        samples = 0
        with (directory / 'samples.bin').open('rb') as source:
            while header := source.read(12):
                self.assertEqual(len(header), 12)
                tokens, actions, won = struct.unpack('iif', header)
                self.assertTrue(0 < actions <= tokens)
                self.assertIn(won, (0., 1.))
                data = source.read(tokens * 68 * 4)
                self.assertEqual(len(data), tokens * 68 * 4)
                self.assertTrue(np.isfinite(np.frombuffer(data, np.float32)).all())
                policy = np.frombuffer(source.read(actions * 4), np.float32)
                self.assertEqual(len(policy), actions)
                self.assertTrue(np.isfinite(policy).all() and (policy >= 0).all())
                self.assertAlmostEqual(float(policy.sum()), 1., places=4)
                samples += 1
        self.assertEqual(samples, sum(r['decisions'] for r in rows))
        return {r['fight_id']: {k:v for k,v in r.items() if k != 'seconds'} for r in rows}

    def test_cpu_scheduler_parity(self):
        common = ['--threads', '2', '--parallel', '4', '--gpu-batch', '4', '--sims', '4']
        legacy = self.play('cpu-ordered', 8, [*common, '--scheduler', 'ordered'])
        local = self.play('cpu-local', 8, [*common, '--scheduler', 'local'])
        self.assertEqual(local, legacy)

    @unittest.skipUnless(os.environ.get('NN_TEST_GPU'), 'explicit GPU model required')
    def test_gpu_local_full_games(self):
        options = ['--scheduler', 'local', '--threads', '2', '--parallel', '16', '--sims', '64',
                   '--gpu-model', os.environ['NN_TEST_GPU'], '--gpu-batch', os.environ.get('NN_TEST_GPU_BATCH', '512'),
                   '--gpu-tokens', '48', '--gpu-wait-us', '100']
        if os.environ.get('NN_TEST_GPU_BANK'):
            options += ['--gpu-bank', os.environ['NN_TEST_GPU_BANK']]
        self.play('gpu-local', 32, options)


if __name__ == '__main__':
    unittest.main()
