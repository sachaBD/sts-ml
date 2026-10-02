"""Offline, read-only probes of the selected Act 2 overworld checkpoint.
Run from repo root: .venv/bin/python -m apps.topology_report.generate
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from agents.overworld.value.core import encode, load_model

DEFAULT = 'runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt'
DATA = 'runs/schema=overworld_v1/date=2026-10-02/id=act2-r02-collect/out'
CARDS = ['body_slam', 'barricade', 'shrug_it_off', 'corruption', 'feel_no_pain',
         'dark_embrace', 'limit_break', 'inflame', 'heavy_blade', 'spot_weakness']


def collect(directory, n):
    names, states, sources = {}, [], []
    for file in sorted(Path(directory).glob('steps-*.parquet')):
        candidate = None
        for row in pq.read_table(file, columns=['record_json']).to_pylist():
            step = json.loads(row['record_json'])
            for state in [step.get('state'), step.get('after')]:
                if not isinstance(state, dict):
                    continue
                for card in state.get('deck', []):
                    names[card['name']] = card['card_id']
                if (candidate is None and state.get('act') == 2 and state['floor'] >= 24
                        and state['map']['paths'] and len(state['map']['paths']) <= 100):
                    candidate = copy.deepcopy(state)
        if candidate is not None and len(states) < n:
            states.append(candidate)
            sources.append(str(file))
        if len(states) == n and all(c in names for c in CARDS):
            break
    if len(states) != n or not all(c in names for c in CARDS):
        raise RuntimeError('Insufficient sample states or missing card names')
    return names, states, sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default=DEFAULT)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default='runs/schema=topology_report/date=2026-10-02/id=act2-selected/out')
    ap.add_argument('--samples', type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = load_model(a.checkpoint)
    if model.KIND != 'run_policy_v1':
        raise ValueError('This report describes v1 only')
    names, states, sources = collect(a.data, a.samples)
    cards = [dict(card_id=names[n], name=n, upgraded=0, misc=0) for n in CARDS]

    @torch.no_grad()
    def logits(ss):
        out = []
        for i in range(0, len(ss), 32):
            part = ss[i:i+32]
            b = encode([(s, []) for s in part], [s['boss'] for s in part])
            out.extend(model(b)[0][:, -1].tolist())
        return np.asarray(out)

    def add(s, *cs):
        t = copy.deepcopy(s)
        t['deck'].extend(copy.deepcopy(cs))
        return t

    base = logits(states)
    single = np.stack([logits([add(s, c) for s in states]) for c in cards], axis=1)
    k = len(cards)
    synergy = np.zeros((len(states), k, k))
    for i in range(k):
        for j in range(i, k):
            both = logits([add(s, cards[i], cards[j]) for s in states])
            synergy[:, i, j] = synergy[:, j, i] = both - single[:, i] - single[:, j] + base
    mean = synergy.mean(0)
    se = synergy.std(0, ddof=1) / np.sqrt(len(states)) if len(states) > 1 else np.zeros_like(mean)
    sig = lambda x: 1 / (1 + np.exp(-x))
    marginal = (sig(single) - sig(base[:, None])) * 100
    hp_grid = list(range(10, 101, 10))
    hp_ss = []
    for pct in hp_grid:
        for s in states:
            t = copy.deepcopy(s)
            t['hp'] = max(1, round(t['max_hp'] * pct / 100))
            hp_ss.append(t)
    hp_values = sig(logits(hp_ss)).reshape(len(hp_grid), -1)
    # Same room counts and occupied floors, different room order.
    route_s = copy.deepcopy(states[0])
    routes = ['NNNNNNNNNREM?MR', 'NNNNNNNNNERM?MR']
    route_values, vectors = [], []
    for rooms in routes:
        t = copy.deepcopy(route_s)
        t['map']['paths'] = [dict(rooms=rooms, first_xs=[0])]
        route_values.append(float(sig(logits([t]))[0]))
        with torch.no_grad():
            vectors.append(model._paths(encode([(t, [])], [t['boss']]))[0, 0])
    permutation = copy.deepcopy(states[0])
    permutation['deck'].reverse()
    permutation_delta = float(logits([permutation])[0] - base[0])
    counts = {}
    for name, p in model.named_parameters():
        group = name.split('.')[0]
        counts[group] = counts.get(group, 0) + p.numel()
    payload = dict(checkpoint=a.checkpoint, sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                   kind=model.KIND, args=model.args, meta={k: ck.get(k) for k in ['target','lam','decay','runs','val_bce']},
                   params=counts, total_params=sum(counts.values()), cards=CARDS, n=len(states), sources=sources,
                   contexts=[dict(boss=s['boss'], floor=s['floor'], hp=s['hp'], max_hp=s['max_hp'],
                                  deck=[c['name'] + ('+' if c['upgraded'] else '') for c in s['deck']]) for s in states],
                   synergy=mean.tolist(), synergy_se=se.tolist(), marginal=marginal.tolist(),
                   hp_grid=hp_grid, hp_mean=hp_values.mean(1).tolist(),
                   hp_sd=hp_values.std(1).tolist(), routes=routes, route_values=route_values,
                   route_vector_maxdiff=float((vectors[0]-vectors[1]).abs().max()),
                   deck_permutation_logit_delta=permutation_delta)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2)
    (out / 'probes.json').write_text(data + '\n')
    template = Path(__file__).with_name('template.html').read_text()
    (out / 'index.html').write_text(template.replace('__DATA__', data.replace('</', '<\\/')))
    print(json.dumps(dict(report=str(out / 'index.html'), n=len(states), params=payload['total_params'],
                          route_maxdiff=payload['route_vector_maxdiff'],
                          strongest_pair=max(((float(mean[i,j]), CARDS[i], CARDS[j]) for i in range(k) for j in range(i+1,k))),
                          weakest_pair=min(((float(mean[i,j]), CARDS[i], CARDS[j]) for i in range(k) for j in range(i+1,k)))), indent=2))


if __name__ == '__main__':
    main()
