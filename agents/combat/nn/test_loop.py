"""Explicit continuation preserves completed work and refuses partial samples."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch

from agents.combat.nn import loop
from agents.combat.nn.model import Net


class ResumeTests(unittest.TestCase):
    def replay(self):
        replay = MagicMock()
        replay.__enter__.return_value = replay
        replay.__len__.return_value = 1
        replay.groups = {'deck': []}
        replay.available = replay.selected = replay.drawn = {'deck': 1}
        return replay

    def prepare(self, root):
        start = root / 'start.json'
        start.write_text(json.dumps({'start': {}}))
        out = root / 'run'
        out.mkdir()
        (out / 'args.json').write_text(json.dumps(dict(games='1', dim='8', layers='1', heads='1', device='cpu', gpu_bank='')))
        (out / 'log.jsonl').write_text(json.dumps(dict(iteration=1)) + '\n')
        net = Net(dim=8, layers=1, heads=1)
        opt = torch.optim.AdamW(net.parameters())
        directory = out / 'iter001'
        directory.mkdir()
        torch.save(dict(net=net.state_dict(), optimizer=opt.state_dict(), spec=net.spec), directory / 'model.pt')
        argv = ['loop', '--start', str(start), '--out', str(out), '--resume-from', '1', '--iterations', '2',
                '--games', '2', '--dim', '8', '--layers', '1', '--heads', '1', '--device', 'cpu', '--steps', '1']
        return out, argv

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, argv = self.prepare(Path(tmp))
            def play(run, model, games, directory, *args, **kwargs):
                self.assertEqual(model.parent.name, 'iter001')
                self.assertEqual(len(games), 2)
                directory.mkdir()
                return [dict(won=True), dict(won=False)]
            with patch('sys.argv', argv), patch.object(loop.Run, 'play', autospec=True, side_effect=play), \
                    patch.object(loop, 'Replay', return_value=self.replay()), patch.object(loop, 'train') as train, \
                    patch.object(loop, 'export'), patch.object(loop, 'export_static'):
                loop.main()
            train.assert_called_once()
            self.assertFalse((out / 'iter000').exists())
            self.assertTrue((out / 'iter001' / 'model.pt').exists())
            saved = torch.load(out / 'iter002' / 'model.pt', weights_only=True)
            self.assertEqual(saved['spec'], dict(dim=8, layers=1, heads=1))
            self.assertIn('rng', saved)
            rows = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['games'], 2)
            self.assertEqual(json.loads((out / 'status.json').read_text())['phase'], 'done')

    def test_export_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = root / 'start.json'
            start.write_text(json.dumps({'start': {}}))
            out = root / 'new-run'
            argv = ['loop', '--start', str(start), '--out', str(out), '--iterations', '1', '--games', '2',
                    '--dim', '8', '--layers', '1', '--heads', '1', '--device', 'cpu', '--steps', '1',
                    '--gpu-batch', '512', '--gpu-bank', '256,128,32', '--gpu-wait-us', '100',
                    '--parallel', '1', '--workers', '2',
                    '--timeout-ms', '321', '--shutdown-timeout-ms', '123']
            def play(run, model, games, directory, *args, **kwargs):
                self.assertEqual(run.gpu_bank, [256, 128, 32])
                self.assertEqual(run.gpu_wait_us, 100)
                self.assertEqual((run.parallel, run.timeout_ms, run.shutdown_timeout_ms), (1, 321, 123))
                directory.mkdir()
                return [dict(won=True), dict(won=False)]
            with patch('sys.argv', argv), patch.object(loop.Run, 'play', autospec=True, side_effect=play), \
                    patch.object(loop, 'Replay', return_value=self.replay()), patch.object(loop, 'train'), \
                    patch.object(loop, 'export'), patch.object(loop, 'export_static') as export:
                loop.main()
            self.assertEqual([c.args[2] for c in export.call_args_list], [512, 256, 128, 32] * 2)
            self.assertEqual([c.args[1].name for c in export.call_args_list],
                             ['model-gpu.onnx', 'model-gpu-b256.onnx', 'model-gpu-b128.onnx', 'model-gpu-b32.onnx'] * 2)

    def test_removed_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            for options in (['--scheduler', 'local'], ['--groups', '2'],
                            ['--replay-cache'], ['--bootstrap', 'old.jsonl']):
                argv = ['loop', '--start', 'unused', '--out', str(out), *options]
                with patch('sys.argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit):
                    loop.main()
                self.assertFalse(out.exists())

    def test_refuse_old_scheduler_run_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, argv = self.prepare(Path(tmp))
            previous = json.loads((out / 'args.json').read_text())
            previous['scheduler'] = 'local'
            (out / 'args.json').write_text(json.dumps(previous))
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'scheduler'):
                loop.main()

    def test_infra_command_has_no_legacy_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = loop.Run(root, gpu_bank=[32], sample_devices=False, play_path=Path('worker'))
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = [json.dumps(dict(fight_id='f', won=True)) + '\n']
            process.returncode = 0
            process.pid = 12345
            with patch.object(loop.subprocess, 'Popen', return_value=process) as popen:
                results = run.play(root / 'model.onnx', [{'fight_id': 'f', 'start': {}}], root / 'play',
                                   4, 1, .5, 2, 'test')
            self.assertEqual(len(results), 1)
            command = popen.call_args.args[0]
            self.assertEqual(command[0], 'worker')
            self.assertIn('--gpu-bank', command)
            for option in ('--scheduler', '--groups', '--gpu-tokens'):
                self.assertNotIn(option, command)

    def test_invalid_bank_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            for bank in ('512', '0', '128,128', 'abc'):
                argv = ['loop', '--start', 'unused', '--out', str(out), '--gpu-batch', '512', '--gpu-bank', bank]
                with patch('sys.argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit):
                    loop.main()
                self.assertFalse(out.exists())

    def test_refuse_changing_bank_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, argv = self.prepare(Path(tmp))
            argv += ['--gpu-bank', '128']
            with patch('sys.argv', argv), self.assertRaises(ValueError):
                loop.main()
            self.assertFalse(list(out.glob('args-before-resume-*')))

    def test_refuse_partial_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, argv = self.prepare(Path(tmp))
            (out / 'iter002').mkdir()
            with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                loop.main()
            self.assertFalse(list(out.glob('args-before-resume-*')))


if __name__ == '__main__':
    unittest.main()
