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

    def play(self, name, count, options, *, append=False, starts=None, sample_path=None):
        directory = self.out / name
        directory.mkdir(exist_ok=append)
        binary = os.environ.get('NN_TEST_BINARY', str(ROOT / 'build/main/agents/combat/nn/nn_play'))
        path = sample_path or directory / 'samples.bin'
        prefix = path.read_bytes() if append else b''
        starts = self.starts[:count] if starts is None else starts
        command = [binary, 'play', os.environ['NN_TEST_MODEL'], str(path),
                   '--rollout-mix', '.5', '--gumbel-scale', '1', *options]
        (directory / 'command.json').write_text(json.dumps(command))
        packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        libraries = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libraries)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        with (directory / 'worker.log').open('w') as errors:
            result = subprocess.run(command, input='\n'.join(starts)+'\n', text=True,
                                    stdout=subprocess.PIPE, stderr=errors, env=env, timeout=90, check=True)
        with (directory / 'results.jsonl').open('a' if append else 'w') as journal:
            journal.write(result.stdout)
        from agents.combat.nn.metrics import last_json
        native_log=(directory/'worker.log').read_text()
        stats=last_json(native_log,'play_stats')
        self.assertEqual(stats['games'],count)
        actors=last_json(native_log,'actors_stats')
        self.assertEqual(actors['actors'], 2)
        self.assertTrue(0 <= actors['actor_cpu_pct'] <= 100.1)
        self.assertTrue(0 <= actors['server_cpu_pct'] <= 100.1)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        expected = {json.loads(line)['fight_id'] for line in starts}
        self.assertEqual(len(rows), count)
        self.assertEqual({r['fight_id'] for r in rows}, expected)
        if any(r['decisions'] for r in rows):
            self.assertGreater(actors['requests'], 0)
        # Journal order is completion order; each range owns exactly its combat's records.
        with path.open('rb') as source:
            self.assertEqual(source.read(len(prefix)), prefix)
            for row in rows:
                span = row['sample_range']
                self.assertEqual(span['file'], path.name)
                self.assertIs(type(span['begin']), int)
                self.assertIs(type(span['end']), int)
                self.assertEqual(span['begin'], source.tell())
                self.assertGreaterEqual(span['end'], span['begin'])
                for _ in range(row['decisions']):
                    header = source.read(12)
                    self.assertEqual(len(header), 12)
                    tokens, actions, won = struct.unpack('<iif', header)
                    self.assertTrue(0 < actions <= tokens)
                    self.assertEqual(won, float(row['won']))
                    data = source.read(tokens * 68 * 4)
                    self.assertEqual(len(data), tokens * 68 * 4)
                    self.assertTrue(np.isfinite(np.frombuffer(data, np.float32)).all())
                    policy = np.frombuffer(source.read(actions * 4), np.float32)
                    self.assertEqual(len(policy), actions)
                    self.assertTrue(np.isfinite(policy).all() and (policy >= 0).all())
                    self.assertAlmostEqual(float(policy.sum()), 1., places=4)
                self.assertEqual(span['end'], source.tell())
            self.assertEqual(source.tell(), path.stat().st_size)
        return {r['fight_id']: {k:v for k,v in r.items() if k not in {'seconds', 'finished_unix', 'sample_range'}} for r in rows}

    def test_cpu_actor_count_parity(self):
        common = ['--threads', '2', '--gpu-batch', '4', '--sims', '4']
        serial = self.play('cpu-one-slot', 8, [*common, '--parallel', '1'])
        concurrent = self.play('cpu-four-slots', 8, [*common, '--parallel', '4'])
        self.assertEqual(serial, concurrent)

    def test_append_and_empty_combats(self):
        options = ['--threads', '2', '--parallel', '4', '--sims', '4']
        self.play('append', 2, options)
        size = (self.out / 'append' / 'samples.bin').stat().st_size
        self.assertGreater(size, 0)
        # An empty deck with no potions has only forced moves, so no searched decisions.
        empty = json.loads(self.starts[0])
        empty['fight_id'] = 'empty'
        empty['start'].update(deck=[], bottled=[-1, -1, -1], relics=[],
                              potion_capacity=0, potions=[], hp=1, max_hp=1)
        first_empty = json.dumps(empty)
        empty['fight_id'] = 'another-empty'
        starts = [first_empty, self.starts[0], json.dumps(empty)]
        rows = self.play('append', len(starts), options, append=True, starts=starts)
        self.assertEqual(rows['empty']['decisions'], 0)
        self.assertEqual(rows['another-empty']['decisions'], 0)
        self.assertGreater((self.out / 'append' / 'samples.bin').stat().st_size, size)
        self.play('empty-file', 1, options, starts=[first_empty])
        self.assertEqual((self.out / 'empty-file' / 'samples.bin').stat().st_size, 0)

    @unittest.skipUnless(Path('/dev/full').exists(), '/dev/full required')
    def test_failed_write_emits_no_result(self):
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.play('failed-write', 1, ['--threads', '2', '--parallel', '1', '--sims', '4'],
                      sample_path=Path('/dev/full'))
        self.assertEqual(error.exception.stdout, '')

    @unittest.skipUnless(os.environ.get('NN_TEST_GPU'), 'explicit GPU model required')
    def test_gpu_actors_full_games(self):
        options = ['--threads', '2', '--parallel', '16', '--sims', '64',
                   '--gpu-model', os.environ['NN_TEST_GPU'], '--gpu-batch', os.environ.get('NN_TEST_GPU_BATCH', '512'),
                   '--gpu-wait-us', '100']
        if os.environ.get('NN_TEST_GPU_BANK'):
            options += ['--gpu-bank', os.environ['NN_TEST_GPU_BANK']]
        self.play('gpu-actors', 32, options)


if __name__ == '__main__':
    unittest.main()
