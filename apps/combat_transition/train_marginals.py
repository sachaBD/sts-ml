#!/usr/bin/env python3
"""Fixed-topology natural vs augmented experiment; see experiments/card-outcomes/README.md."""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from apps.combat_transition.train import load, batch, Baseline, metrics
from sts_combat_rl.run import RUNS
from sts_combat_rl.topology.combat_outcome_v1 import CombatOutcomeV1, hp_bin, BINS
from sts_combat_rl.topology.combat_outcome_v2 import CombatOutcomeV2
import hashlib, tomllib

KINDS = {k.KIND: k for k in (CombatOutcomeV1, CombatOutcomeV2)}
FROZEN = Path(__file__).resolve().parents[2]/'topology/combat_outcome/FROZEN.tsv'


def topology(path):
    """Topology spec (topology/combat_outcome/*.toml): kind + constructor args. Frozen once trained (FROZEN.tsv)."""
    if path is None:
        return dict(name='default', sha256=None, kind='combat_outcome_v1', args=dict(width=16, hidden=32, dropout=.3))
    raw = Path(path).read_bytes(); sha = hashlib.sha256(raw).hexdigest(); spec = tomllib.loads(raw.decode())
    name = Path(path).stem
    for line in (FROZEN.read_text().splitlines() if FROZEN.exists() else []):
        n, h = line.split('\t')[:2]
        if n == name and h != sha:
            raise ValueError(f'Topology {name} is frozen (trained) but its file changed; make a new spec')
    return dict(name=name, sha256=sha, kind=spec['kind'], args=spec['args'])


def runpath(rid):
    schema, date, name = rid.split('/')
    return RUNS / f'schema={schema}' / f'date={date}' / f'id={name}' / 'out'


def parquet_rows(file):
    # Do not materialise repeated post states or group descriptions for every fight.
    columns = ['stage', 'group_seed', 'encounter', 'variant', 'seed', 'seed_index',
               'donor_run_seed', 'kind', 'battle_final_hp', 'won', 'start_hp', 'pre']
    cached = {}
    for part in pq.ParquetFile(file).iter_batches(batch_size=1024, columns=columns):
        for row in part.to_pylist():
            if 'kind' not in row and row['stage'] == 'easy':
                row['kind'] = 'easy'  # legacy easy-pilot schema
            key = (row['stage'], row['group_seed'], row['variant'], row['encounter'])
            pre = row['pre']
            if key in cached and cached[key] == pre:
                row['pre'] = cached[key]
            else:
                cached[key] = pre
            yield row


def dataset(natural, marginals, eval_marginals=None):
    references = set(marginals if eval_marginals is None else eval_marginals)
    if not references or not references.issubset(set(marginals)):
        raise ValueError("Evaluation runs must be a nonempty subset of cumulative data runs")
    rows, excluded = load(natural)
    excluded['event_outside_scope'] = sum(r['category'] == 'event' for r in rows)
    rows = [r for r in rows if r['category'] != 'event']
    for r in rows:
        b = r['run_seed'] % 10
        # Bucket 8 is held out everywhere: donors in synthetic development must not be natural training.
        r.update(domain='natural', split='dev' if b in (0, 1, 8) else 'stop' if b == 9 else 'train',
                 cluster=f"natural:{r['run_seed']}")
    seen = set()
    for rid in marginals:
        meta = json.loads((runpath(rid).parent/'run.json').read_text())
        if meta['status'] != 'done':
            raise ValueError(f'Incomplete input run: {rid}')
        for file in sorted(runpath(rid).glob('*.parquet')):
            for r in parquet_rows(file):
                key = (r['stage'], r['group_seed'], r['encounter'], r['variant'], r['seed'])
                if key in seen:
                    raise ValueError(f'Duplicate synthetic observation across inputs: {key}')
                seen.add(key)
                donor = r.get('donor_run_seed')
                b = (donor if donor is not None else r['group_seed']) % 10
                if donor is not None:
                    if b not in (4, 5, 6, 7, 8, 9):
                        raise ValueError('Unexpected donor outside training/stop source cohort')
                    split = 'dev' if b == 8 else 'stop' if b == 9 else 'train'
                else:
                    if b in (2, 3):
                        continue
                    split = 'dev' if b in (0, 1) else 'stop' if b == 9 else 'train'
                if split in ('dev', 'stop') and rid not in references:
                    continue  # Freeze evaluation AND early-stop cohorts at stage 1.
                r.update(final_hp=r['battle_final_hp'], category=r['kind'], domain='synthetic', split=split,
                         cluster=f"donor:{donor}" if donor is not None else f"group:{r['group_seed']}")
                rows.append(r)
    return rows, excluded


def infer(net, data, device):
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(data[1]), 1024):
            b = {k: v[start:start+1024].to(device) for k, v in data[0].items()}
            w, h = net(b)
            out.append((w.sigmoid().cpu(), h.softmax(-1).cpu()))
    return torch.cat([x[0] for x in out]), torch.cat([x[1] for x in out])


def fit(net, rows, stoprows, enc, device, epochs, baseline=False):
    train, stop = batch(rows, enc, 'cpu'), batch(stoprows, enc, 'cpu')
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=.03 if baseline else float(os.environ.get('CARD_OUTCOME_LR', '.003')), weight_decay=0 if baseline else .1)
    # Equal category mass; in mixed training, equal natural/synthetic mass within each category.
    cells = Counter((r['category'], r['domain']) for r in rows)
    domains = Counter(c for c, d in cells)
    weights = torch.tensor([1/(cells[r['category'], r['domain']]*domains[r['category']]) for r in rows])
    best, saved, bad, best_epoch = float('inf'), None, 0, 0
    # Optional paired auxiliary loss (CARD_OUTCOME_PAIR_W, default 0 = off): for synthetic variant rows, squared error of the
    # predicted expected-score difference (vs the base deck, same group/encounter/fight seed) against the observed difference.
    pair_w = 0. if baseline else float(os.environ.get('CARD_OUTCOME_PAIR_W', '0'))
    if pair_w:
        base_of = {(r['stage'], r['group_seed'], r['encounter'], r['seed']): i for i, r in enumerate(rows)
                   if r['domain'] == 'synthetic' and r['variant'] == 0}
        partner = torch.tensor([base_of.get((r['stage'], r['group_seed'], r['encounter'], r['seed']), -1)
                                if r['domain'] == 'synthetic' and r['variant'] else -1 for r in rows])
        bin_centers = torch.tensor([3.+5*i for i in range(BINS-1)]+[103.], device=device)
        score_of = lambda w, h: w.sigmoid()*(h.softmax(-1)*bin_centers).sum(-1)
        print(json.dumps(dict(pair_w=pair_w, pairs=int((partner >= 0).sum()))), flush=True)
    for epoch in range(epochs):
        net.train()
        # Bound each epoch rather than letting cheap easy rows dictate work.
        ids = torch.multinomial(weights, min(len(rows), 32768), replacement=True)
        for ii in ids.split(512):
            b = {k: v[ii].to(device) for k, v in train[0].items()}
            won, hp = train[1][ii].to(device), train[2][ii].to(device)
            w, h = net(b)
            loss = F.binary_cross_entropy_with_logits(w, won)
            loss += (F.cross_entropy(h, hp_bin(hp), reduction='none') * won).mean()
            if baseline:
                loss += net.penalty()/len(rows)
            if pair_w:
                jj = ii[partner[ii] >= 0]
                if len(jj):
                    kk = partner[jj]
                    pb = {k: torch.cat([v[jj], v[kk]]).to(device) for k, v in train[0].items()}
                    pw, ph = net(pb); sc = score_of(pw, ph); n = len(jj)
                    obs = (train[1]*train[2])[torch.cat([jj, kk])].to(device)
                    loss += pair_w*(((sc[:n]-sc[n:])-(obs[:n]-obs[n:]))**2).mean()/100
            opt.zero_grad(); loss.backward(); opt.step()
        p, h = infer(net, stop, device)
        loss = F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), stop[1], reduction='none')
        loss += -h.gather(1, hp_bin(stop[2])[:, None]).squeeze(1).clamp_min(1e-9).log()*stop[1]
        cellvalues = defaultdict(list)
        for r, x in zip(stoprows, loss.tolist()):
            cellvalues[r['category'], r['domain']].append(x)
        cats = defaultdict(list)
        for (c, d), xs in cellvalues.items():
            cats[c].append(sum(xs)/len(xs))
        score = sum(sum(xs)/len(xs) for xs in cats.values())/len(cats)
        print(json.dumps(dict(epoch=epoch+1, stop_joint_nll=score)), flush=True)
        if score < best-1e-4:
            best, bad, best_epoch = score, 0, epoch+1
            saved = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(saved)
    return best_epoch


def paired(rows, p, h, centers):
    cells = defaultdict(lambda: defaultdict(list))
    mean = p*(h*centers).sum(-1)
    for i,r in enumerate(rows):
        if r['domain'] != 'synthetic': continue
        key = (r['stage'], r['group_seed'], r['encounter'])
        real = r['start_hp']-r['final_hp']+30*(not r['won'])
        pred = r['start_hp']-mean[i].item()+30*(1-p[i].item())
        cells[key][r['variant']].append((real,pred,r['seed']))
    result = defaultdict(list)
    for (stage, group, encounter), variants in cells.items():
        if 0 not in variants: continue
        avg = lambda xs: (sum(x[0] for x in xs)/len(xs),sum(x[1] for x in xs)/len(xs))
        base = avg(variants[0])
        for v,xs in variants.items():
            if v == 0: continue
            if {x[2] for x in xs} != {x[2] for x in variants[0]}:
                raise ValueError('Unmatched seeds in paired evaluation')
            a,b = avg(xs); delta,pred = a-base[0],b-base[1]
            result[encounter].append((abs(pred-delta),abs(delta)))
    return {k: dict(pairs=len(v), score_delta_mae=sum(x[0] for x in v)/len(v),
                    zero_effect_mae=sum(x[1] for x in v)/len(v)) for k,v in result.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('natural'); ap.add_argument('--marginals', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--topology',help='topology/combat_outcome/<name>.toml (default: legacy w16/h32/d.3 combat_outcome_v1)')
    ap.add_argument('--arms',nargs='+',default=['natural','augmented'],choices=['natural','augmented'])
    ap.add_argument('--eval-marginals',nargs='+',help='Fixed first-stage runs supplying synthetic dev/early-stop rows')
    args=ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(int(os.environ.get('CARD_OUTCOME_SEED', '0'))); random.seed(int(os.environ.get('CARD_OUTCOME_SEED', '0')))
    args.out.mkdir(parents=True,exist_ok=True)
    if args.epochs < 1: ap.error('--epochs must be positive')
    rows, excluded=dataset(args.natural,args.marginals,args.eval_marginals)
    print(f'Loaded {len(rows)} rows', flush=True)
    counts=Counter((r['domain'],r['split'],r['category']) for r in rows)
    enc={e:i+1 for i,e in enumerate(sorted({r['encounter'] for r in rows if r['split']=='train'}))}
    dev=[r for r in rows if r['split']=='dev']; test=batch(dev,enc,'cpu')
    reports={}
    topo=topology(args.topology)
    for arm in args.arms:
        torch.manual_seed(int(os.environ.get('CARD_OUTCOME_SEED', '0')))
        train=[r for r in rows if r['split']=='train' and (arm=='augmented' or r['domain']=='natural')]
        stop=[r for r in rows if r['split']=='stop' and (arm=='augmented' or r['domain']=='natural')]
        if not train or not stop or not dev: raise ValueError('Empty split')
        base=Baseline(len(enc))
        fit(base,train,stop,enc,args.device,args.epochs,True)
        net=KINDS[topo['kind']](encounters=len(enc)+1,**topo['args'])
        with torch.no_grad():
            net.enc_linear.copy_((base.g[None]+base.e).cpu())
            net.win_out.weight.zero_(); net.win_out.bias.zero_(); net.hp_out.weight.zero_(); net.hp_out.bias.zero_()
        epoch=fit(net,train,stop,enc,args.device,args.epochs)
        over=[r['final_hp'] for r in train if r['won'] and r['final_hp']>100]
        centers=torch.tensor([3.+5*i for i in range(BINS-1)]+[sum(over)/len(over) if over else 103.])
        p,h=infer(net,test,args.device)
        per={}
        for domain,cat in sorted({(r['domain'],r['category']) for r in dev}):
            ii=torch.tensor([i for i,r in enumerate(dev) if (r['domain'],r['category'])==(domain,cat)])
            per[f'{domain}/{cat}']=metrics(p[ii],h[ii],test[1][ii],test[2][ii],centers)
        for domain,e in sorted({(r['domain'],r['encounter']) for r in dev}):
            ii=torch.tensor([i for i,r in enumerate(dev) if (r['domain'],r['encounter'])==(domain,e)])
            per[f'{domain}/encounter/{e}']=metrics(p[ii],h[ii],test[1][ii],test[2][ii],centers)
        reports[arm]=dict(topology=topo,best_epoch=epoch,metrics=per,paired=paired(dev,p,h,centers))
        with (args.out/f'{arm}-predictions.jsonl').open('w') as f:
            for i,r in enumerate(dev):
                item={k:r.get(k) for k in ('domain','category','encounter','cluster','stage','group_seed','variant','seed','run_seed')}
                item.update(won=bool(r['won']), final_hp=r['final_hp'], start_hp=r.get('start_hp',r['pre']['hp']),
                            p_win=p[i].item(), mean_hp_if_win=(h[i]*centers).sum().item())
                f.write(json.dumps(item)+'\n')
        torch.save(dict(kind=net.KIND,args=net.args,state_dict={k:v.cpu() for k,v in net.state_dict().items()},
                        encounters=enc,centers=centers.tolist()),args.out/f'{arm}.pt')
        (args.out/'report.json').write_text(json.dumps(dict(counts={str(k):v for k,v in counts.items()},
            excluded=excluded,inputs=[args.natural,*args.marginals],eval_marginals=args.eval_marginals or args.marginals,results=reports),indent=2))
    (args.out/'report.md').write_text('# Overnight fixed-topology comparison\n\nSee report.json for per-category outcomes and paired card-effect MAE versus zero effect.\n\n'
        'One training seed; descriptive only. Synthetic development holds out donor bucket 8 (natural bucket 8 also excluded from training); easy holds out generated-group buckets 0–1. Natural development includes buckets 0–1 and 8. No confidence intervals or policy-win claims. '
        'Both arms use the same category-balanced joint loss; these are new baselines, not exact reproductions of iteration 1.\n')
    if topo['sha256']:
        with FROZEN.open('a') as f: f.write(f"{topo['name']}\t{topo['sha256']}\t{args.out}\n")
    if len(reports)==2:
      with (args.out/'report.md').open('a') as f:
        f.write('\n| Domain/category | Fights | Natural Brier | Augmented Brier | Natural HP MAE | Augmented HP MAE |\n|---|---:|---:|---:|---:|---:|\n')
        for k,a in reports['natural']['metrics'].items():
            if '/encounter/' in k: continue
            b=reports['augmented']['metrics'][k]
            f.write(f"| {k} | {a['fights']} | {a['brier']:.4f} | {b['brier']:.4f} | {a.get('hp_mae',float('nan')):.2f} | {b.get('hp_mae',float('nan')):.2f} |\n")
        f.write('\n| Encounter | Card pairs | Zero-effect MAE | Natural-model delta MAE | Augmented-model delta MAE |\n|---|---:|---:|---:|---:|\n')
        for k,a in reports['natural']['paired'].items():
            b=reports['augmented']['paired'][k]
            f.write(f"| {k} | {a['pairs']} | {a['zero_effect_mae']:.3f} | {a['score_delta_mae']:.3f} | {b['score_delta_mae']:.3f} |\n")
    (args.out/'summary.json').write_text(json.dumps(dict(inputs=[args.natural,*args.marginals],results=reports)))

if __name__=='__main__': main()
