"""Do repeated searches on the same card decision agree? usage: agreement.py DATA.jsonl...
Per decision and rep: option values Q = mean over rollouts of score (HP at the end of act 1, 0 if dead) and of clear.
Paired (common random numbers: rollout i shares futures across options) vs unpaired SE of option - skip differences."""
import collections, json, math, statistics as st, sys

reps = collections.defaultdict(dict)
for f in sys.argv[1:]:
    for m in map(json.loads, open(f)):
        if m['status'] == 'ok':
            reps[(m['seed'], m['decision'])][m['job']['rollout_seed']] = m
agree, top_gap, paired_se, unpaired_se, spread, n_opts, corr = [], [], [], [], [], [], []
for key, rs in reps.items():
    if len(rs) < 2: continue
    Q = []
    for m in rs.values():
        R = m['rollouts']; skip = [r['hp'] for r in R[-1]]
        Q.append([st.mean(r['hp'] for r in o) for o in R])
        for o in R[:-1]:
            d = [a['hp'] - b for a, b in zip(o, skip)]
            paired_se.append(st.stdev(d) / math.sqrt(len(d)))
            unpaired_se.append(math.sqrt(st.variance([a['hp'] for a in o]) / len(o) + st.variance(skip) / len(skip)))
    a, b = Q[0], Q[1]
    agree.append(a.index(max(a)) == b.index(max(b)))
    spread.append(max(a) - min(a)); n_opts.append(len(a))
    if len(a) > 2 and st.pstdev(a) > 0 and st.pstdev(b) > 0: corr.append(st.correlation(a, b))
print(f'decisions with 2 reps: {len(agree)}; options per decision: {st.mean(n_opts):.2f}')
print(f'best option agrees across reps: {st.mean(agree):.2f} (chance ~{st.mean(1 / k for k in n_opts):.2f})')
print(f'Pearson r of option values across reps: mean {st.mean(corr):.2f} (n={len(corr)})')
print(f'spread of option values (max - min, HP-score): median {st.median(spread):.2f}, mean {st.mean(spread):.2f}')
print(f'SE of option - skip: paired (CRN) median {st.median(paired_se):.2f} vs unpaired {st.median(unpaired_se):.2f} '
      f'-> CRN variance reduction x{(st.median(unpaired_se) / st.median(paired_se)) ** 2:.1f}')
