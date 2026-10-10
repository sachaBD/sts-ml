"""Fixed-step hybrid replay trainer. Input manifests are supplied by the workflow app.

Only the online-half fight sampler changes from Phase5: geometric recency weights.
Targets, concentration policy weighting, architecture, AdamW and optimizer resume stay unchanged.
No validation-based checkpoint selection. Output must be a new directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from .data import load_split, merge
from .model import CONTRACT, NAMES, PolicyValue
from .recency import ReplayPool
from .train import evaluate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(manifest, output, device='cpu'):
    cfg = json.loads(Path(manifest).read_text())
    required = {'init', 'init_sha256', 'split', 'anchor', 'online', 'update', 'half_life', 'steps', 'rng_seed'}
    if not required <= cfg.keys():
        raise ValueError(f'missing manifest keys: {required - cfg.keys()}')
    if cfg['steps'] <= 0 or not isinstance(cfg['steps'], int):
        raise ValueError('positive integer steps required')
    output = Path(output)
    if output.exists():
        raise ValueError('output already exists; partial training requires explicit review, never overwrite')
    if sha(cfg['init']) != cfg['init_sha256']:
        raise ValueError('initial checkpoint hash mismatch')
    split = load_split(cfg['split'])
    for spec in cfg['anchor'] + cfg['online']:
        if sha(spec['path']) != spec['sha256']:
            raise ValueError(f'shard hash mismatch: {spec["path"]}')
    anchor = ReplayPool(cfg['anchor'], split, cfg['update'])
    online = ReplayPool(cfg['online'], split, cfg['update'], cfg['half_life'])
    if anchor.ids & online.ids:
        raise ValueError('anchor/online fight overlap')
    ck = torch.load(cfg['init'], map_location='cpu', weights_only=False)
    if ck['contract'] != CONTRACT or 'optimizer_state_dict' not in ck:
        raise ValueError('incompatible checkpoint or missing AdamW state')
    torch.set_num_threads(2)
    torch.manual_seed(cfg['rng_seed'])
    net = PolicyValue(ck['width'], ck['value_activation'])
    net.load_state_dict(ck['state_dict']); net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=.01)
    opt.load_state_dict(ck['optimizer_state_dict'])
    for g in opt.param_groups:
        g.update(lr=3e-4, weight_decay=.01)
    before = max(int(s['step']) for s in opt.state.values())
    rngs = [np.random.default_rng(np.random.SeedSequence([cfg['rng_seed'], cfg['update'], h])) for h in (0, 1)]
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(cfg, indent=2))
    start = time.time(); sums = np.zeros(2); logs = []
    for step in range(1, cfg['steps']+1):
        batch = merge([anchor.sample(rngs[0], 32), online.sample(rngs[1], 32)])
        net.train()
        v, p, _, n, npol = evaluate(net, batch, device, flat_policy_weighting=True)
        loss = v/n + p/max(npol, 1)
        if not torch.isfinite(loss):
            raise ValueError(f'nonfinite loss at step {step}')
        opt.zero_grad(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step(); sums += (v.item()/n, p.item()/max(npol, 1))
        if step % 100 == 0 or step == cfg['steps']:
            count = (step-1) % 100 + 1
            row = dict(step=step, total_steps=cfg['steps'], value_loss=sums[0]/count,
                       policy_loss=sums[1]/count, last_grad_norm=float(norm), seconds=time.time()-start)
            print(json.dumps(row), flush=True); logs.append(row); sums[:] = 0
    after = max(int(s['step']) for s in opt.state.values())
    assert after == before + cfg['steps']
    assert sum(anchor.realized.values()) == sum(online.realized.values()) == 32*cfg['steps']
    cpu = PolicyValue(net.width, net.value_activation)
    cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()}); cpu.eval()
    torch.save(dict(contract=CONTRACT, width=net.width, value_activation=net.value_activation,
                    state_dict=cpu.state_dict(), optimizer_state_dict=opt.state_dict()), output/'model.pt')
    # Independent export/check RNG; do not count verification rows as training exposure.
    check = online.shards[0]
    example = check.batch(check.rows[:2])[0]
    cpu.export(example, output/'model.onnx')
    import onnxruntime as ort
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output/'model.onnx'), sess_options=opts, providers=['CPUExecutionProvider'])
    x = check.batch(check.rows[:32])[0]
    got = session.run(None, {n:x[n].numpy() for n in NAMES})
    with torch.inference_mode(): ref = cpu(*(x[n] for n in NAMES))
    diff = float(max(np.max(np.abs(a-b.numpy())) for a,b in zip(got, ref)))
    if diff > 1e-3:
        raise ValueError(f'ONNX parity failed: {diff}')
    report = dict(config=cfg, adam_step_before=before, adam_step_after=after, log=logs,
                  anchor_fights=len(anchor.ids), online_fights=len(online.ids),
                  anchor_draws=sum(anchor.realized.values()), online_draws=sum(online.realized.values()),
                  expected_online_generation_share=online.expected_generation_share(),
                  realized_online_generation_draws=dict(online.generation_draws),
                  online_fight_draws=dict(online.realized), onnx_max_abs_diff=diff,
                  files_sha256={p.name:sha(p) for p in output.glob('model.*')}, seconds=time.time()-start)
    (output/'train.json').write_text(json.dumps(report, indent=2))
    # Published last: a checkpoint with no completion marker is never a collection model.
    (output/'complete.json').write_text(json.dumps({'model_sha256':sha(output/'model.pt')}))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args(); train(a.manifest, a.out, a.device)


if __name__ == '__main__':
    main()
