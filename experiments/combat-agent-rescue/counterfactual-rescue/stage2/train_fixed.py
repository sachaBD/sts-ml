#!/usr/bin/env python3
"""Isolated fixed-step trainer for Stage 2 (stage2/PROTOCOL.md). Does not alter default agents.combat.pv.train.

Every optimizer step uses exactly 32 OLD + 32 NEW states. Each half samples a fight uniformly (with replacement),
then one state uniformly within that fight. OLD and NEW use independent RNG streams seeded from
SeedSequence([0, update, 0/1]), so both arms draw identical OLD rows whatever their NEW pool sizes.
Loss = value/n + policy/n_policy with concentration (flat-policy) weighting, as the original
`pv.train --flat-policy-weighting` with unit weights. AdamW moments resume from --init; lr 3e-4, wd 0.01, clip 1.
No validation or checkpoint selection: the checkpoint after the last step is saved and exported.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))  # run from the repository root
from agents.combat.pv.data import Shard, merge, load_split  # noqa: E402
from agents.combat.pv.model import PolicyValue, CONTRACT, NAMES  # noqa: E402
from agents.combat.pv.train import evaluate  # noqa: E402

LOCKED = dict(steps=1000, half=32, lr=3e-4, weight_decay=0.01, grad_clip=1.0)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')


class Pool:
    """Training rows of several shards, indexed by fight."""

    def __init__(self, paths, split):
        self.shards = [Shard(Path(p), False, split) for p in paths]
        self.fights = []  # (fight_id, shard index, row indices)
        for i, s in enumerate(self.shards):
            ids = s.fight_ids[s.rows]
            for fid in np.unique(ids):
                self.fights.append((str(fid), i, s.rows[ids == fid]))
        if not self.fights: raise ValueError('empty training half')
        self.states = int(sum(len(r) for _, _, r in self.fights))

    def sample(self, rng, n):
        picks = rng.integers(len(self.fights), size=n)
        by = {}
        for k in picks:
            _, i, rows = self.fights[k]
            by.setdefault(i, []).append(rows[rng.integers(len(rows))])
        return merge([self.shards[i].batch(np.array(idx)) for i, idx in sorted(by.items())])


def run(a):
    split = load_split(a.split)
    old, new = Pool(a.old, split), Pool(a.new, split)
    old_ids, new_ids = {f[0] for f in old.fights}, {f[0] for f in new.fights}
    if len(old_ids) != len(old.fights) or len(new_ids) != len(new.fights) or old_ids & new_ids:
        raise ValueError('duplicate fight id within/between halves')
    if a.expect_old_fights is not None and len(old_ids) != a.expect_old_fights:
        raise ValueError(f'old half has {len(old_ids)} fights, expected {a.expect_old_fights}')
    if a.expect_new_fights is not None and len(new_ids) != a.expect_new_fights:
        raise ValueError(f'new half has {len(new_ids)} fights, expected {a.expect_new_fights}')
    torch.manual_seed(0)
    seeds = {'old': [0, a.update, 0], 'new': [0, a.update, 1]}
    rng_old, rng_new = (np.random.default_rng(np.random.SeedSequence(seeds[k])) for k in ('old', 'new'))
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    if ck['contract'] != CONTRACT: raise ValueError('contract mismatch')
    if 'optimizer_state_dict' not in ck: raise ValueError('init has no optimizer state')
    net = PolicyValue(ck['width'], ck.get('value_activation', 'softplus')); net.load_state_dict(ck['state_dict']); net.to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=LOCKED['lr'], weight_decay=LOCKED['weight_decay'])
    opt.load_state_dict(ck['optimizer_state_dict'])
    for g in opt.param_groups: g.update(lr=LOCKED['lr'], weight_decay=LOCKED['weight_decay'])
    adam_step_before = max(int(s['step']) for s in opt.state_dict()['state'].values())
    counts = {'old': 0, 'new': 0, 'steps': 0}; sums = np.zeros(3); log = []; t0 = time.time()
    for step in range(a.steps):
        ob, nb = old.sample(rng_old, LOCKED['half']), new.sample(rng_new, LOCKED['half'])
        if len(ob[1]) != LOCKED['half'] or len(nb[1]) != LOCKED['half']: raise AssertionError('half size')
        b = merge([ob, nb]); net.train()
        v, q, _, n, npol = evaluate(net, b, a.device, flat_policy_weighting=True)
        loss = v / n + q / max(npol, 1)
        if not torch.isfinite(loss): raise ValueError(f'non-finite loss at step {step}')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), LOCKED['grad_clip']); opt.step()
        counts['old'] += len(ob[1]); counts['new'] += len(nb[1]); counts['steps'] += 1
        sums += (v.item() / n, q.item() / max(npol, 1), loss.item())
        if (step + 1) % 100 == 0 or step + 1 == a.steps:
            row = {'step': step + 1, 'value_loss': sums[0] / ((step % 100) + 1), 'policy_loss': sums[1] / ((step % 100) + 1),
                   'loss': sums[2] / ((step % 100) + 1), 'seconds': time.time() - t0}
            log.append(row); print(json.dumps(row), flush=True); sums[:] = 0
    want = {'old': LOCKED['half'] * a.steps, 'new': LOCKED['half'] * a.steps, 'steps': a.steps}
    if counts != want: raise AssertionError(counts)
    adam_step_after = max(int(s['step']) for s in opt.state_dict()['state'].values())
    if adam_step_after != adam_step_before + a.steps: raise AssertionError('optimizer moments did not resume/advance')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({'contract': CONTRACT, 'width': net.width, 'value_activation': net.value_activation,
                'state_dict': net.state_dict(), 'optimizer_state_dict': opt.state_dict()}, out / 'model.pt')
    saved_at = stamp()
    # Export a CPU copy of the final checkpoint; check ONNX vs torch on held-out OLD rows.
    cpu = PolicyValue(net.width, net.value_activation); cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()}); cpu.eval()
    check_rng = np.random.default_rng(np.random.SeedSequence([0, a.update, 2]))
    example = old.sample(check_rng, 2)[0]
    cpu.export(example, out / 'model.onnx')
    exported_at = stamp()
    parity = onnx_parity(cpu, out / 'model.onnx', merge([old.sample(check_rng, 16), new.sample(check_rng, 16)])[0])
    if parity > 1e-3: raise AssertionError(f'ONNX/torch parity {parity}')
    files = {p.name: sha(p) for p in sorted(out.glob('model.*'))}
    report = {'counts': counts, 'locked': LOCKED, 'update': a.update, 'rng_seed_sequences': seeds, 'torch_seed': 0,
              'init': str(a.init), 'init_sha256': sha(a.init), 'adam_step_before': adam_step_before, 'adam_step_after': adam_step_after,
              'old_fights': len(old_ids), 'old_states': old.states, 'new_fights': len(new_ids), 'new_states': new.states,
              'old_shards': [str(p) for p in a.old], 'new_shards': [str(p) for p in a.new], 'log': log,
              'onnx_max_abs_diff': parity, 'files_sha256': files, 'saved_at': saved_at, 'exported_at': exported_at,
              'train_seconds': time.time() - t0, 'test_override_steps': a.steps != LOCKED['steps']}
    (out / 'train.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def onnx_parity(net, path, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    got = sess.run(None, {n: inputs[n].numpy() for n in NAMES})
    with torch.no_grad(): ref = net(*(inputs[n] for n in NAMES))
    return float(max(np.abs(g - r.numpy()).max() for g, r in zip(got, ref)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old', nargs='+', required=True); p.add_argument('--new', nargs='+', required=True)
    p.add_argument('--split', required=True); p.add_argument('--init', required=True); p.add_argument('--out', required=True)
    p.add_argument('--update', type=int, required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--expect-old-fights', type=int); p.add_argument('--expect-new-fights', type=int)
    p.add_argument('--test-steps', type=int, help='TEST ONLY: override the locked 1000 steps')
    a = p.parse_args()
    a.steps = LOCKED['steps'] if a.test_steps is None else a.test_steps
    run(a)


if __name__ == '__main__': main()
