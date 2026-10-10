"""S3 confirmation for the human-deck corpus: frozen learned2k vs MCTS20k on reserved, never-played final seeds.

Predeclared (PLAN.md): at most --max-decks provisional-graduated decks, in graduation order (first graduation update,
ties by deck-id hash), 30 reserved final seeds each. Plus Barricade A forgetting check on its 100 single-deck monitor
seeds against the cached MCTS reference. Missing pairs are reported, never filled by extending the deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pyarrow.parquet as pq

from apps.run_rl.combat_loop import play, read_results, write_starts
from apps.run_rl.single_deck import wilson

A_RUN = Path('runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-v1/out')


def first_graduation(rec):
    ks = [int(m.group(1)) for r in rec['reasons'] for m in [re.match(r'k=(\d+): \w+ -> graduated', r)] if m]
    return min(ks) if ks else 10**9


def paired(ids, base, cand):
    pairs = [(base[f]['fight']['won'], cand[f]['fight']['won']) for f in ids
             if base.get(f, {}).get('status') == 'completed' and cand.get(f, {}).get('status') == 'completed']
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='controller out dir')
    p.add_argument('--model', type=Path, required=True, help='frozen model dir (controller freeze)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-decks', type=int, default=12)
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    state = json.loads((a.run / 'state.json').read_text())
    worker = a.run / 'frozen/pv_worker'
    grad = [(first_graduation(r), hashlib.sha256(d.encode()).hexdigest(), d) for d, r in state['decks'].items()
            if r['status'] == 'graduated']
    chosen = [d for _, _, d in sorted(grad)][:a.max_decks]
    final = [r for r in pq.ParquetFile(a.run / 'final-reserved.parquet').read().to_pylist() if r['deck_id'] in chosen]
    if any(sum(r['deck_id'] == d for r in final) != 30 for d in chosen):
        raise ValueError('expected 30 reserved final starts per chosen deck')
    json.dump(dict(chosen=chosen, model=str(a.model.resolve()), graduated_total=len(grad), update=state['k']),
              open(out / 'selection.json', 'w'), indent=1)
    write_starts(final, out / 'final-starts.parquet')
    learned = play(out, 'learned', out / 'final-starts.parquet', a.model, worker, 2000, a.workers)
    mcts = play(out, 'mcts', out / 'final-starts.parquet', None, worker, 20000, a.workers, teacher=True)
    L, M = read_results(learned), read_results(mcts)
    rows, diffs = [], []
    for d in chosen:
        ids = [r['fight_id'] for r in final if r['deck_id'] == d]
        pr = paired(ids, M, L)
        n = len(pr)
        lw, mw = sum(y for _, y in pr), sum(x for x, _ in pr)
        rows.append(dict(deck=d, n=n, learned=lw, mcts=mw, learned_only=sum(y and not x for x, y in pr),
                         mcts_only=sum(x and not y for x, y in pr), missing=len(ids) - n))
        if n:
            diffs.append((lw - mw) / n)
    g = len(diffs)
    gap = sum(diffs) / g if g else None
    se = math.sqrt(sum((x - gap) ** 2 for x in diffs) / (g * (g - 1))) if g > 1 else None
    # A forgetting check: learned2k on A's 100 single-deck monitor seeds vs cached MCTS20k reference.
    a_starts = A_RUN / 'monitor.parquet'
    a_learned = play(out, 'a-monitor', a_starts, a.model, worker, 2000, a.workers)
    AL, AM = read_results(a_learned), read_results(A_RUN / 'reference-monitor')
    apr = paired([r['fight_id'] for r in pq.ParquetFile(a_starts).read().to_pylist()], AM, AL)
    summary = dict(decks=rows, deck_macro_gap=gap, deck_macro_gap_se=se,
                   learned_total=sum(r['learned'] for r in rows), mcts_total=sum(r['mcts'] for r in rows),
                   pairs=sum(r['n'] for r in rows), a_check=dict(n=len(apr), learned=sum(y for _, y in apr),
                                                                  mcts=sum(x for x, _ in apr)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    lines = ['# S3 confirmation: frozen learned2k vs MCTS20k on reserved final seeds', '',
             f'Model `{a.model}` (controller update {state["k"]}). {len(chosen)} of {len(grad)} provisional-graduated decks '
             f'(graduation order, max {a.max_decks}); 30 never-played final seeds each. Selected-corpus confirmation, '
             'not performance across all decks.', '',
             '| deck | pairs | learned | MCTS | learned-only | MCTS-only | missing |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["deck"][:8]} | {r["n"]} | {r["learned"]} | {r["mcts"]} | {r["learned_only"]} | {r["mcts_only"]} | {r["missing"]} |')
    if summary['pairs']:
        lt, mt, n = summary['learned_total'], summary['mcts_total'], summary['pairs']
        lo, hi = wilson(lt, n)
        lines += ['', f'Pooled: learned {lt}/{n} ({100*lt/n:.1f}%, Wilson {100*lo:.1f}–{100*hi:.1f}%) vs MCTS {mt}/{n} ({100*mt/n:.1f}%).',
                  f'Deck-macro paired gap {100*gap:+.1f} pts' + (f' ± {100*se:.1f} (1 SE across decks, n={g})' if se else '') + '.']
    ac = summary['a_check']
    lines += ['', f'Barricade A forgetting check (100 single-deck monitor seeds, not untouched): learned {ac["learned"]}/{ac["n"]} '
                  f'vs MCTS {ac["mcts"]}/{ac["n"]} (single-deck specialist scored 99/100 here).']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
