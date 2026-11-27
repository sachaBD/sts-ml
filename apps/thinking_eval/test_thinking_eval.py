import os
import tempfile
import unittest
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from agents.combat.nn.model import ModelSpec, build_model
from agents.combat.nn.pytorch_to_onnx import Cpu, Deployment, ModelExporter
from apps.thinking_eval.eval import (
    build_summary_and_report,
    collect_positions,
    collector_executable,
    evaluate_checkpoint,
    extract_targets,
    load_collected,
    topk_capture,
)


def _sample_fight(seed=7):
    return {
        'fight_id': f'test:boss:deck_{seed}:0',
        'start': {
            'seed': seed,
            'ascension': 20,
            'act': 2,
            'floor': 33,
            'encounter': 39,
            'cur_room': 6,
            'last_room': 1,
            'burning_elite_buff': -1,
            'hp': 60,
            'max_hp': 80,
            'gold': 50,
            'misc_rng': {'counter': 0, 'seed0': 12345, 'seed1': 67890},
            'potion_rng': {'counter': 0, 'seed0': 12345, 'seed1': 67890},
            'potion_capacity': 3,
            'potions': [1, 1, 1],
            'relics': [{'id': 21, 'data': 0}],
            'deck': [
                {'id': 15, 'upgraded': False, 'misc': 0},
                {'id': 104, 'upgraded': False, 'misc': 0},
                {'id': 104, 'upgraded': False, 'misc': 0},
                {'id': 104, 'upgraded': False, 'misc': 0},
                {'id': 104, 'upgraded': False, 'misc': 0},
                {'id': 321, 'upgraded': False, 'misc': 0},
                {'id': 321, 'upgraded': False, 'misc': 0},
                {'id': 321, 'upgraded': False, 'misc': 0},
                {'id': 321, 'upgraded': False, 'misc': 0},
                {'id': 321, 'upgraded': False, 'misc': 0},
            ],
            'bottled': [-1, -1, -1],
        },
    }


class ThinkingEvalTest(unittest.TestCase):
    def test_topk_capture_bounds_and_oracle_monotonicity(self):
        rng = np.random.default_rng(11)
        target = rng.exponential(scale=0.05, size=200)
        oracle = topk_capture(target, target, fracs=(0.05, 0.10, 0.20, 0.30), resamples=50)
        noise = topk_capture(rng.normal(size=200), target, fracs=(0.05, 0.10, 0.20, 0.30), resamples=50)
        shares = [oracle[k]['share'] for k in ('top_5pct', 'top_10pct', 'top_20pct', 'top_30pct')]
        self.assertEqual(shares, sorted(shares))
        self.assertGreater(oracle['top_10pct']['share'], noise['top_10pct']['share'])
        for k in ('top_5pct', 'top_10pct', 'top_20pct', 'top_30pct'):
            self.assertTrue(0.0 <= oracle[k]['share'] <= 1.0 + 1e-9)

    def test_end_to_end_cpu_collect_and_analyze_invariants(self):
        binary = collector_executable()
        if not binary.exists():
            self.skipTest(f'{binary} not built')

        torch.manual_seed(0)
        spec = ModelSpec(kind='inventory_v1', dim=32, layers=1, heads=4, value='outcome', auxiliary=True)
        net = build_model(spec).eval()
        exporter = ModelExporter(spec, (Deployment(tokens=48, batches=(4,), lanes=1),), Cpu(lanes=1, batch=8),
                                 mixed_precision=False)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            model_dir = tmp_dir / 'export'
            model_dir.mkdir()
            exporter.export(net, model_dir)
            ckpt_path = tmp_dir / 'model.pt'
            torch.save({'spec': spec.__dict__, 'net': net.state_dict()}, ckpt_path)

            fights = [_sample_fight(seed=s) for s in (7, 13, 29)]
            out_dir = tmp_dir / 'eval_out'
            collect_positions(
                binary,
                model_dir,
                fights,
                out_dir,
                sims=16,
                particles=2,
                considered_actions=8,
                horizon_probe=True,
                probe_sims=12,
                probe_particles=2,
                grade_particles=2,
                actors=2,
                slots=4,
                cpu=True,
            )

            rows, states, orders = load_collected(out_dir)
            self.assertEqual(len(rows), 3)
            for row, order in zip(rows, orders):
                self.assertGreaterEqual(len(order), 2)
                self.assertEqual(len(row['draws']), 2)
                self.assertNotEqual(row['draws'][0]['world_draw'], row['draws'][1]['world_draw'])
                for rd in row['horizon_probe']['rounds']:
                    if rd['selected'] == rd['prior']:
                        self.assertEqual(rd['gain_h1_a'], 0.0)
                        self.assertEqual(rd['gain_h1_b'], 0.0)
                        self.assertEqual(rd['gain_end_a'], 0.0)
                        self.assertEqual(rd['gain_end_b'], 0.0)

            targets = extract_targets(rows)
            self.assertTrue(np.all(targets['vos_b'] >= -1e-9))
            ev = evaluate_checkpoint(ckpt_path, states, orders, device='cpu')
            # Zero-initialized ThinkingHead output layer must emit exact zeros before training
            np.testing.assert_allclose(ev['thinking'], 0.0, atol=1e-7)
            summary, report = build_summary_and_report(rows, targets, {'init': ev})
            self.assertEqual(summary['positions'], 3)
            self.assertIn('ThinkingHead Held-Out State Evaluation', report)


if __name__ == '__main__':
    unittest.main()
