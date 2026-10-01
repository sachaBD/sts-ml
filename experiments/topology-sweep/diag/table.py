"""Sweep table: per (topology, lr) mean and [min..max] over training seeds.
Supplementary dev metrics come from report.json (Brier) and the predictions (card dMSE vs zero, unbiased).
With --hs DIR, also scores the hs16 predictions (DIR/<run>-augmented-predictions.jsonl) at the pair level:
  dMSE (unbiased, per-fight) and R2 = explained fraction of true (noise-corrected) pair-effect variance.
usage: table.py PREFIX [--hs DIR]"""
import collections, glob, json, re, statistics as st, sys

def pair_stats(path):
    """Per category: dMSE (mean pred^2 - 2 pred obs over per-seed obs) and noise-corrected R2 on per-pair means."""
    g = collections.defaultdict(dict)
    for r in map(json.loads, open(path)):
        if r['domain'] != 'synthetic': continue
        g[(r['category'], r['cluster'], r['group_seed'], r['encounter'], r['seed'])][r['variant']] = (
            r['final_hp'] if r['won'] else 0, r['p_win'] * r['mean_hp_if_win'])
    pairs = collections.defaultdict(list)
    for (cat, cl, gs, enc, seed), v in g.items():
        if 0 not in v: continue
        for k, x in v.items():
            if k: pairs[(cat, cl, gs, enc, k)].append((x[0] - v[0][0], x[1] - v[0][1]))
    out = {}
    for cat in {k[0] for k in pairs}:
        L = [s for k, s in pairs.items() if k[0] == cat and len(s) >= 2]
        d = st.mean(p * p - 2 * p * o for s in L for o, p in s)
        m = [st.mean(o for o, _ in s) for s in L]; pm = [st.mean(p for _, p in s) for s in L]
        noise = st.mean(st.variance([o for o, _ in s]) / len(s) for s in L)
        true_var = st.pvariance(m) - noise
        mse = st.mean((a - b) ** 2 for a, b in zip(m, pm)) - noise
        out[cat] = dict(dmse=d, r2=1 - mse / true_var if true_var > 0 else float('nan'), true_sd=max(true_var, 0) ** .5,
                        pairs=len(L), seeds=st.mean(len(s) for s in L))
    return out

prefix = sys.argv[1]; hs = sys.argv[sys.argv.index('--hs') + 1] if '--hs' in sys.argv else None
cells = collections.defaultdict(list)
for d in sorted(glob.glob(f'runs/schema=combat_outcome_v1/date=*/id={prefix}-*/out')):
    run = d.split('id=')[1].split('/')[0]
    m = re.match(re.escape(prefix) + r'-(.+)-lr([.\d]+)-s(\d+)$', run)
    if not m or not glob.glob(d + '/report.json'): continue
    a = json.load(open(d + '/report.json'))['results']['augmented']
    row = {'epoch': a['best_epoch']}
    for k in ('natural/boss', 'natural/elite', 'synthetic/boss', 'synthetic/elite'):
        row['Brier ' + k] = a['metrics'][k]['brier']
    for cat, s in pair_stats(d + '/augmented-predictions.jsonl').items():
        row['supp dMSE ' + cat] = s['dmse']
    if hs:
        f = f'{hs}/{run}-augmented-predictions.jsonl'
        if glob.glob(f):
            for cat, s in pair_stats(f).items():
                row['hs dMSE ' + cat] = s['dmse']; row['hs R2 ' + cat] = s['r2']
    cells[(m[1], m[2])].append(row)
keys = sorted({k for rows in cells.values() for r in rows for k in r}, key=lambda k: (k.split()[0], k))
print('| topology | lr | n | ' + ' | '.join(keys) + ' |\n|' + '---|' * (3 + len(keys)))
for (t, lr), rows in sorted(cells.items()):
    fmt = lambda k: (lambda v: f'{st.mean(v):.3g} [{min(v):.3g}..{max(v):.3g}]' if v else '')([r[k] for r in rows if k in r])
    print(f'| {t} | {lr} | {len(rows)} | ' + ' | '.join(fmt(k) for k in keys) + ' |')
