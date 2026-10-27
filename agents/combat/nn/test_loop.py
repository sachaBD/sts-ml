"""Explicit continuation preserves completed work and refuses partial samples."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from agents.combat.nn import loop
from agents.combat.nn.model import Net, WIDTH


class ResumeTests(unittest.TestCase):
    def prepare(self, root):
        start = root / 'start.json'
        start.write_text(json.dumps({'start': {}}))
        out = root / 'run'
        out.mkdir()
        (out / 'args.json').write_text(json.dumps(dict(games='1', dim='8', layers='1', heads='1', device='cpu')))
        (out / 'log.jsonl').write_text(json.dumps(dict(iteration=1)) + '\n')
        net = Net(dim=8, layers=1, heads=1)
        opt = torch.optim.AdamW(net.parameters())
        directory = out / 'iter001'
        directory.mkdir()
        torch.save(dict(net=net.state_dict(), optimizer=opt.state_dict(), spec=net.spec), directory / 'model.pt')
        argv = ['loop', '--start', str(start), '--out', str(out), '--resume-from', '1', '--iterations', '2',
                '--games', '2', '--dim', '8', '--layers', '1', '--heads', '1', '--device', 'cpu', '--steps', '1']
        return out, argv

    def test_resume_without_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, argv = self.prepare(Path(tmp))
            def play(run, model, games, directory, *args, **kwargs):
                self.assertEqual(model.parent.name, 'iter001')
                self.assertEqual(len(games), 2)
                directory.mkdir()
                return [dict(won=True), dict(won=False)]
            data = (torch.zeros(1, 2, WIDTH), torch.zeros(1, 2), torch.ones(1))
            with patch('sys.argv', argv), patch.object(loop.Run, 'play', autospec=True, side_effect=play), \
                    patch.object(loop, 'read', return_value=data), patch.object(loop, 'train') as train, \
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
                    '--scheduler', 'local', '--parallel', '1', '--workers', '2',
                    '--timeout-ms', '321', '--shutdown-timeout-ms', '123']
            def play(run, model, games, directory, *args, **kwargs):
                self.assertEqual(run.gpu_bank, [256, 128, 32])
                self.assertEqual(run.gpu_wait_us, 100)
                self.assertEqual((run.scheduler, run.parallel, run.timeout_ms, run.shutdown_timeout_ms), ('local', 1, 321, 123))
                directory.mkdir()
                return [dict(won=True), dict(won=False)]
            data = (torch.zeros(1, 2, WIDTH), torch.zeros(1, 2), torch.ones(1))
            with patch('sys.argv', argv), patch.object(loop.Run, 'play', autospec=True, side_effect=play), \
                    patch.object(loop, 'read', return_value=data), patch.object(loop, 'train'), \
                    patch.object(loop, 'export'), patch.object(loop, 'export_static') as export:
                loop.main()
            self.assertEqual([c.args[2] for c in export.call_args_list], [512, 256, 128, 32] * 2)
            self.assertEqual([c.args[1].name for c in export.call_args_list],
                             ['model-gpu.onnx', 'model-gpu-b256.onnx', 'model-gpu-b128.onnx', 'model-gpu-b32.onnx'] * 2)

    def test_invalid_bank_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            for bank in ('512', '0', '128,128', 'abc'):
                argv = ['loop', '--start', 'unused', '--out', str(out), '--gpu-batch', '512', '--gpu-bank', bank]
                with patch('sys.argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit):
                    loop.main()
                self.assertFalse(out.exists())

    def test_refuse_changing_legacy_bank_on_resume(self):
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
