"""Fixed-budget corpus training. Frozen inventory -> family-first replay -> final-epoch export.

Reuses the canonical Shard, model and loss. Only sampled unique rows are held in memory;
all fight/state draws retain multiplicity. No validation-based checkpoint selection.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

from .corpus_replay import Fight, plan_epoch
from .data import Shard, merge
from .model import CONTRACT, NAMES, PolicyValue
from .train import evaluate


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ReplayData:
    def __init__(self, specs):
        self.specs, self.index, self.fights = specs, {}, []
        for i, spec in enumerate(specs):
            if sha(spec['path']) != spec['sha256']:
                raise ValueError(f'shard hash mismatch: {spec["path"]}')
            ids = pq.ParquetFile(spec['path']).read(columns=['fight_id'])['fight_id'].to_pylist()
            grouped = defaultdict(list)
            for row, fid in enumerate(ids):
                grouped[fid].append(row)
            if set(grouped) != set(spec['fights']):
                raise ValueError('shard inventory does not exactly match recorded fights')
            for fid, rows in grouped.items():
                if fid in self.index:
                    raise ValueError(f'duplicate fight across shards: {fid}')
                meta = spec['fights'][fid]
                self.index[fid] = (i, np.asarray(rows))
                self.fights.append(Fight(fid, meta['family_id'], meta['source'], meta['round'], len(rows)))
        if not self.fights:
            raise ValueError('empty replay')

    def batches(self, plan, batch_size=64):
        draws = plan.mixed_state_draws()
        requested = defaultdict(set)
        locations = []
        for _, _, fid, offset in draws:
            shard, rows = self.index[fid]
            row = int(rows[offset])
            requested[shard].add(row)
            locations.append((shard, row))
        loaded, remap = {}, {}
        for i, wanted in requested.items():
            selected = sorted(wanted)
            split = {fid: 'train' for fid in self.specs[i]['fights']}
            loaded[i] = Shard(self.specs[i]['path'], False, split, row_indices=selected)
            remap[i] = {r: k for k, r in enumerate(selected)}
        for start in range(0, len(locations), batch_size):
            groups = defaultdict(list)
            for shard, row in locations[start:start + batch_size]:
                groups[shard].append(remap[shard][row])
            yield merge([loaded[i].batch(np.asarray(rows)) for i, rows in groups.items()])


def train(manifest, output, device='cuda', smoke_steps=None):
    cfg = json.loads(Path(manifest).read_text())
    output = Path(output)
    if output.exists():
        raise ValueError('training output already exists; review partial attempt before retry')
    if sha(cfg['init']) != cfg['init_sha256']:
        raise ValueError('checkpoint hash mismatch')
    ck = torch.load(cfg['init'], map_location='cpu', weights_only=False)
    if ck['contract'] != CONTRACT or ck['width'] != 64 or ck['value_activation'] != 'sigmoid':
        raise ValueError('incompatible checkpoint')
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    net = PolicyValue(ck['width'], ck['value_activation']).to(device)
    net.load_state_dict(ck['state_dict'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=.01)
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    for group in optimizer.param_groups:
        group.update(lr=3e-4, weight_decay=.01)
    before = max(int(s['step']) for s in optimizer.state.values())
    data = ReplayData(cfg['shards'])
    families = [SimpleNamespace(**f) for f in cfg['families']]
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text(json.dumps(cfg, indent=2))
    step, started, reports, example = 0, time.monotonic(), [], None
    for epoch in range(1 if smoke_steps else 3):
        seed = cfg['seed'] + epoch
        plan = plan_epoch(families, data.fights, cfg['round'], seed)
        report = dict(epoch=epoch + 1, slots=len(plan.slots), states=plan.state_count,
                      counts={f'{family}/{source}': count for (family, source), count in plan.counts().items()},
                      unique_fights=len({s.fight_id for s in plan.slots}))
        print(json.dumps(dict(event='epoch-plan', **report)), flush=True)
        totals = np.zeros(2)
        net.train()
        for batch in data.batches(plan):
            if example is None:
                example = {k: v[:2].clone() for k, v in batch[0].items()}
            v, p, _, n, npol = evaluate(net, batch, device, flat_policy_weighting=True)
            loss = v / n + p / max(npol, 1)
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite loss: step {step}')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            totals += (v.item() / n, p.item() / max(npol, 1))
            if step % 100 == 0 or smoke_steps:
                print(json.dumps(dict(event='train', step=step, total_steps=smoke_steps or 9000,
                                      epoch=epoch+1, seconds=time.monotonic()-started)), flush=True)
            if smoke_steps and step >= smoke_steps:
                break
        report.update(loss_sums=totals.tolist())
        reports.append(report)
        gc.collect()
    expected = smoke_steps or 9000
    after = max(int(s['step']) for s in optimizer.state.values())
    if step != expected or after != before + expected:
        raise ValueError('optimizer budget mismatch')
    cpu = PolicyValue(ck['width'], ck['value_activation'])
    cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
    cpu.eval()
    torch.save(dict(contract=CONTRACT, width=ck['width'], value_activation=ck['value_activation'],
                    state_dict=cpu.state_dict(), optimizer_state_dict=optimizer.state_dict()), output / 'model.pt')
    cpu.export(example, output / 'model.onnx')
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    got = session.run(None, {k: example[k].numpy() for k in NAMES})
    with torch.inference_mode():
        ref = cpu(*(example[k] for k in NAMES))
    error = float(max(np.max(np.abs(a-b.numpy())) for a, b in zip(got, ref)))
    if error > 1e-3:
        raise ValueError(f'ONNX parity failed: {error}')
    summary = dict(steps=step, epochs=reports, onnx_max_error=error, numpy=np.__version__,
                   seconds=time.monotonic()-started, smoke=bool(smoke_steps),
                   checkpoint_sha256=sha(output / 'model.pt'), onnx_sha256=sha(output / 'model.onnx'))
    (output / 'complete.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-steps', type=int)
    args = parser.parse_args()
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 3000:
        parser.error('smoke steps must be 1..3000')
    train(args.manifest, args.out, args.device, args.smoke_steps)
