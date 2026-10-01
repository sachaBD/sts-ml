"""Evaluate saved train_marginals checkpoints (<arm>.pt) on synthetic dev rows of other card_marginals runs.

eval_marginals.py NATURAL_RUN --marginals RUN... --checkpoints A.pt B.pt ... --out DIR
Writes DIR/<checkpoint tag>-predictions.jsonl (same format as train_marginals predictions). Tag = parent run id + arm.
Only held-out (dev) synthetic rows are scored; no training happens here.
"""
import argparse
import json
from pathlib import Path

import torch

from apps.combat_transition.train import batch
from apps.combat_transition.train_marginals import KINDS, dataset, infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('natural'); ap.add_argument('--marginals', nargs='+', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, _ = dataset(args.natural, args.marginals, args.marginals)
    dev = [r for r in rows if r['split'] == 'dev' and r['domain'] == 'synthetic']
    print(f'{len(dev)} synthetic dev rows', flush=True)
    for ck in args.checkpoints:
        c = torch.load(ck, map_location='cpu')
        net = KINDS[c['kind']](**c['args']); net.load_state_dict(c['state_dict']); net.to(args.device)
        enc = c['encounters']; keep = [r for r in dev if r['encounter'] in enc]
        p, h = infer(net, batch(keep, enc, 'cpu'), args.device)
        centers = torch.tensor(c['centers'])
        tag = Path(ck).resolve().parents[1].name.removeprefix('id=') + '-' + Path(ck).stem
        with (args.out/f'{tag}-predictions.jsonl').open('w') as f:
            for i, r in enumerate(keep):
                item = {k: r.get(k) for k in ('domain', 'category', 'encounter', 'cluster', 'stage', 'group_seed', 'variant',
                                              'seed', 'run_seed')}
                item.update(won=bool(r['won']), final_hp=r['final_hp'], p_win=p[i].item(),
                            mean_hp_if_win=(h[i]*centers).sum().item())
                f.write(json.dumps(item)+'\n')
        print(f'{tag}: {len(keep)} rows', flush=True)


if __name__ == '__main__':
    main()
