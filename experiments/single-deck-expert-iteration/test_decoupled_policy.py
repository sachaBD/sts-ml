"""Checks for the evaluator-decoupled composite (synthetic small networks; no gameplay)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))
import decoupled_policy as D  # noqa: E402
from agents.combat.pv.model import NAMES, PolicyValue  # noqa: E402


def inputs(batch=3, actions=4, cards=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = {'context': torch.randn(batch, 65, generator=g)}
    for n, w, k in [('cards', 18, cards), ('monsters', 29, 2), ('potions', 19, 2), ('relics', 4, 3), ('actions', 261, actions)]:
        t = torch.rand(batch, k, w, generator=g)
        t[..., 0] = torch.randint(1, 30, (batch, k), generator=g).float()
        x[n] = t
    x['actions'][..., 0] = torch.randint(0, 5, (batch, actions), generator=g).float()
    x['actions'][..., 1] = 0
    x['actions'][..., 260] = 1
    x['actions'][0, -1] = 0  # one padded action slot
    return x


def nets():
    torch.manual_seed(0); a = PolicyValue(8, 'sigmoid').eval()
    torch.manual_seed(1); b = PolicyValue(8, 'sigmoid').eval()
    return a, b


def test_composite_routes_value_and_policy():
    a, b = nets(); x = inputs()
    with torch.no_grad():
        va, la = a(*(x[n] for n in NAMES)); vb, lb = b(*(x[n] for n in NAMES))
        v, l = D.Composite(a, b)(*(x[n] for n in NAMES)); v2, l2 = D.Composite(a, a)(*(x[n] for n in NAMES))
    assert torch.equal(v, va) and torch.equal(l, lb) and not torch.equal(l, la)
    assert torch.equal(v2, va) and torch.equal(l2, la)
    assert (l[0, -1] == -1e9) and torch.equal(l[0, -1], lb[0, -1])  # padding mask unchanged


def test_no_dropout_or_batchnorm():
    a, _ = nets()
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) for m in a.modules())


def test_export_schema_dynamic_shapes_and_fail_closed():
    a, b = nets()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'model.onnx'
        D.export(D.Composite(a, b), inputs(2, 3, 2), path, {'composite': 'test'})
        assert D.check_onnx(path)['composite'] == 'test'
        s = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        for batch, actions, cards in [(1, 2, 2), (5, 9, 7), (64, 3, 30)]:  # dynamic batch/token/action counts
            x = inputs(batch, actions, cards, seed=batch)
            v, l = s.run(None, {n: x[n].numpy() for n in NAMES})
            with torch.no_grad(): tv, _ = a(*(x[n] for n in NAMES)); _, tl = b(*(x[n] for n in NAMES))
            assert np.abs(v - tv.numpy()).max() < 1e-4 and np.abs(l - tl.numpy())[x['actions'][..., 260].numpy() != 0].max() < 1e-4
        for mutate in ['contract', 'input', 'output']:
            g = onnx.load(path)
            if mutate == 'contract':
                del g.metadata_props[:]
            elif mutate == 'input':
                g.graph.input[0].name = 'ctx'
            else:
                g.graph.output[0].name = 'v'
            bad = Path(tmp) / f'bad-{mutate}.onnx'; onnx.save(g, bad)
            try:
                D.check_onnx(bad); raise AssertionError(f'{mutate} tamper not rejected')
            except ValueError:
                pass


def test_generate_skips_used_seeds():
    from apps.run_rl.single_deck import generate
    base = dict(fight_id='x', seed_kind='x', start=dict(seed=0, misc_rng=None, potion_rng=None))
    first = generate(base, D.EXP, 'dev', 3, set())
    used = {first[0]['start']['seed']}
    again = generate(base, D.EXP, 'dev', 3, set(used))
    assert again[0]['start']['seed'] not in used and again[1]['start']['seed'] == first[1]['start']['seed']
    assert first[0]['fight_id'] == f'{D.EXP}:dev:0'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'): fn(); print('ok', name)
