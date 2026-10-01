"""Objective model-space metrics on hs16 (16 fight seeds per (deck, fight) state; bucket-8 donors, never trained).
No new compute: reads existing hs16-eval prediction files.

Per state s (group, encounter, variant): measured win rate w_s over n=16 seeds; the model's p_s is deterministic per state.
  Win-rate RMSE (noise-corrected): sqrt( mean (p-w)^2 - mean w(1-w)/(n-1) )   [binomial noise of w removed, unbiased]
  Win-rate R2: 1 - corrected MSE / true between-state variance (var(w) - noise)
  ECE: |mean p - mean w| over p-deciles, weighted by count (w noise averages out within bins)
Decisions per (group, encounter): choose among base (skip) + 4 candidate cards by expected score = P(win)*E[HP|win].
  Value = measured mean score (16 seeds; HP if won else 0) of the chosen option. Unbiased for any rule that does not look
  at outcomes. Reference rules: always skip, random (mean of options), oracle-split (choose on seeds 0-7, score on 8-15;
  the model is scored on seeds 8-15 too in that column). Regret = oracle-split - rule, both on seeds 8-15.
usage: objective.py LABEL=GLOB [LABEL=GLOB ...]   (GLOB over hs16-eval/*-predictions.jsonl; seeds averaged per label)"""
import collections, glob, json, math, random, statistics as st, sys

def load(path):
    S = collections.defaultdict(list); P = {}
    for r in map(json.loads, open(path)):
        k = (r['category'], r['encounter'], r['group_seed'], r['variant'])
        S[k].append((r['seed'], r['won'], r['final_hp'] if r['won'] else 0))
        P[k] = (r['p_win'], r['p_win'] * r['mean_hp_if_win'])
    return S, P

def metrics(S, P, cat):
    ks = [k for k in S if k[0] == cat]
    w = {k: st.mean(x[1] for x in S[k]) for k in ks}; n = {k: len(S[k]) for k in ks}
    noise = st.mean(w[k] * (1 - w[k]) / (n[k] - 1) for k in ks)
    mse = st.mean((P[k][0] - w[k]) ** 2 for k in ks) - noise
    tv = st.pvariance([w[k] for k in ks]) - noise
    bins = collections.defaultdict(list)
    for k in ks: bins[min(int(P[k][0] * 10), 9)].append(k)
    ece = sum(len(v) * abs(st.mean(P[k][0] for k in v) - st.mean(w[k] for k in v)) for v in bins.values()) / len(ks)
    # decisions
    groups = collections.defaultdict(dict)
    for k in ks: groups[k[:3]][k[3]] = k
    val = collections.defaultdict(list)
    for g, opts in groups.items():
        if 0 not in opts or len(opts) < 2: continue
        full = {v: st.mean(x[2] for x in S[k]) for v, k in opts.items()}
        seeds = sorted({x[0] for k in opts.values() for x in S[k]}); half = set(seeds[::2])  # shared fight seeds split in halves
        A = {v: st.mean(x[2] for x in S[k] if x[0] in half) for v, k in opts.items()}
        B = {v: st.mean(x[2] for x in S[k] if x[0] not in half) for v, k in opts.items()}
        pick = max(opts, key=lambda v: P[opts[v]][1]); orc = max(A, key=A.get)
        val['model'].append(full[pick]); val['skip'].append(full[0]); val['random'].append(st.mean(full.values()))
        val['model_B'].append(B[pick]); val['oracle_B'].append(B[orc]); val['agree_oracle_full'].append(pick == max(full, key=full.get))
        val['pwin_model'].append(st.mean(x[1] for x in S[opts[pick]])); val['pwin_skip'].append(st.mean(x[1] for x in S[opts[0]]))
    d = {k: st.mean(v) for k, v in val.items()}
    return dict(states=len(ks), decisions=len(val['model']), rmse=math.sqrt(max(mse, 0)), r2=1 - mse / tv, true_sd=math.sqrt(max(tv, 0)),
                ece=ece, mean_p=st.mean(P[k][0] for k in ks), mean_w=st.mean(w.values()), **d)

rows = collections.defaultdict(list)
for arg in sys.argv[1:]:
    label, pat = arg.split('=', 1)
    for f in sorted(glob.glob(pat)):
        S, P = load(f)
        for cat in ('boss', 'elite', 'hard'):
            rows[(label, cat)].append(metrics(S, P, cat))
cols = ['rmse', 'r2', 'ece', 'mean_p', 'mean_w', 'model', 'skip', 'random', 'model_B', 'oracle_B', 'agree_oracle_full', 'pwin_model', 'pwin_skip']
print('| model | cat | n seeds | states/decisions | ' + ' | '.join(cols) + ' |\n|' + '---|' * (4 + len(cols)))
for (label, cat), L in rows.items():
    fmt = lambda c: (lambda v: f'{st.mean(v):.3f}' + (f' [{min(v):.3f}..{max(v):.3f}]' if len(v) > 1 and c in ('rmse', 'r2', 'ece', 'model') else ''))([m[c] for m in L])
    print(f'| {label} | {cat} | {len(L)} | {L[0]["states"]}/{L[0]["decisions"]} | ' + ' | '.join(fmt(c) for c in cols) + ' |')
print(f'\ntrue between-state win-rate sd: ' + ', '.join(f'{c} {rows[(sys.argv[1].split("=")[0], c)][0]["true_sd"]:.3f}' for c in ('boss', 'elite', 'hard')))
