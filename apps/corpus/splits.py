"""Freeze family-level held-out / training-pool manifests for the human-deck corpus (PLAN.md, Splits).

Held out (never screened or trained): r01 validation families + original-431 benchmark families,
minus explicitly consumed controls. Training pool: r01 train-side human starts whose family is not held out,
plus consumed controls. Fails closed on any unexpected overlap.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq

from apps.run_rl.combat_loop import fingerprint, sha, write_starts

R01 = Path('runs/schema=combat_v4/date=2026-10-05/id=human-combat-r01/out')
BENCH = Path('runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet')
LIBRARY = Path('runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1/out')
# Already trained-on or screened members of the 431 benchmark (PLAN.md): released into the training pool.
CONSUMED = ['7a9ada48-a4d6-41fc-a85b-0147618df00a', '00205aa1-6004-4ab0-8600-655bddec017e', '478ffcb8', '82a04d39',
            '02d4f101', '35a58920', 'ca29d1e5', '351e9475']


def human(rows):
    out, seen = [], set()
    for r in rows:
        if r['seed_kind'] == 'human' and r['deck_id'] not in seen:
            seen.add(r['deck_id']); out.append(r)
    return out


def resolve(prefixes, rows):
    ids = {}
    for p in prefixes:
        hit = {r['deck_id'] for r in rows if r['deck_id'].startswith(p)}
        if len(hit) != 1:
            raise ValueError(f'consumed prefix {p} matched {len(hit)} decks')
        ids[p] = hit.pop()
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source = human(pq.read_table(R01 / 'source/starts.parquet').to_pylist())
    bench = human(pq.read_table(BENCH).to_pylist())
    split = json.loads((R01 / 'split.json').read_text())
    library = json.loads((LIBRARY / 'manifest.json').read_text())
    consumed = resolve(CONSUMED, bench)
    consumed_fams = {fingerprint(r) for r in bench if r['deck_id'] in consumed.values()}

    val_rows = [r for r in source if split.get(r['fight_id']) == 'val']
    held = {fingerprint(r) for r in val_rows} | {fingerprint(r) for r in bench}
    held -= consumed_fams
    # Consumed controls must not also be r01 validation families.
    if consumed_fams & {fingerprint(r) for r in val_rows}:
        raise ValueError('consumed control overlaps r01 validation family')

    # r01 'train' plus r01-unmapped source decks; anything whose family is held out is excluded.
    pool = [r for r in source if split.get(r['fight_id']) != 'val' and fingerprint(r) not in held]
    pool_ids = {r['deck_id'] for r in pool}
    pool += [r for r in bench if r['deck_id'] in consumed.values() and r['deck_id'] not in pool_ids]
    # One row per family in the pool (first by deck id) so duplicate families cannot leak across splits.
    by_family = {}
    for r in sorted(pool, key=lambda r: r['deck_id']):
        by_family.setdefault(fingerprint(r), r)
    pool = list(by_family.values())
    if {fingerprint(r) for r in pool} & held:
        raise ValueError('pool/held-out family overlap')

    lib = {m['deck_id'] for m in library}
    protected = sorted(d for d in lib if any(r['deck_id'] == d for r in source + bench)
                       and fingerprint(next(r for r in source + bench if r['deck_id'] == d)) in held)
    write_starts(pool, a.out / 'pool.parquet')
    summary = dict(pool_decks=len(pool), heldout_families=len(held), consumed_controls=consumed,
                   training_protected_library=protected,
                   library_in_pool=sorted(d for d in lib if d in {r['deck_id'] for r in pool}),
                   pool_by_bucket=dict(collections.Counter(bucket_of(r) for r in pool)),
                   inputs={str(x): sha(x) for x in (R01 / 'source/starts.parquet', R01 / 'split.json', BENCH)})
    (a.out / 'heldout-families.json').write_text(json.dumps(sorted(held)))
    (a.out / 'splits-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'inputs'}, indent=2))


def bucket_of(r):
    from apps.human_champ.library import bucket
    return bucket(r)


if __name__ == '__main__':
    main()
