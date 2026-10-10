"""Build the corpus controller's decks.json (+ filtered teacher source tables) from screen results.

Pre-registered rules (PLAN.md):
- Eligible: all 20 screen fights completed, mean MCTS seconds <= --max-seconds, 1-19/20 MCTS wins, not
  training-protected, in the frozen pool. 20/20 decks are appended last (low priority); 0/20 are listed, not used.
- Wave 0 (pilot): A (Barricade) + 7 decks with 3-17/20 wins and mean <= --pilot-seconds: one per remaining
  archetype bucket first, then the rest by |p-0.5|, ties by deck-id hash.
- Later waves: round-robin over buckets, within a bucket by |p-0.5| then hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from agents.combat.pv.data import WORKER
from apps.human_champ.library import bucket, results

A = '7a9ada48-a4d6-41fc-a85b-0147618df00a'
LIBRARY = Path('runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1')
REFS = [Path('runs/schema=combat_v4/date=2026-10-05/id=single-deck-mcts20'),
        Path('runs/schema=combat_v4/date=2026-10-05/id=strength-deck-mcts20')]
BUCKETS = ['block', 'demon_form', 'exhaust', 'strength', 'mixed']


def h(x):
    return hashlib.sha256(x.encode()).hexdigest()


def screened(play_dirs):
    """deck_id -> list of (fight_id, play_dir, record) from completed and failed screen plays."""
    out = {}
    for d in play_dirs:
        for fid, r in results(d / 'results.jsonl').items():
            deck = fid.split(':')[1] if fid.count(':') >= 2 else None
            if deck is None:
                raise ValueError(f'unexpected fight id {fid}')
            out.setdefault(deck, []).append((fid, d, r))
    return out


def encodable(row, worker):
    """True if the pv encoder replays one screen fight of the deck (catches unsupported cards, e.g. RAGNAROK/ALPHA)."""
    from agents.combat.pv.data import collect
    from apps.corpus.controller import subset
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'src'
        subset([Path(r) for r in row['teacher_runs']], row['teacher_fight_ids'][:1], src)
        try:
            collect([src / 'fights-0.parquet'], [src / 'search-0.parquet'], [39], Path(tmp) / 'out', worker)
        except (RuntimeError, subprocess.CalledProcessError):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--splits-summary', type=Path, required=True)
    p.add_argument('--screen', type=Path, nargs='+', required=True, help='screen run dirs (apps.corpus.screen)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, default=60)
    p.add_argument('--pilot-seconds', type=float, default=30)
    p.add_argument('--pilot', type=int, default=8)
    p.add_argument('--wave-size', type=int, default=8)
    p.add_argument('--check-encodable', action='store_true', help='encode 1 screen fight per eligible deck; skip decks the pv encoder rejects')
    p.add_argument('--worker', type=Path, help='pv_worker for --check-encodable (default: pv.data.WORKER)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pool = {r['deck_id']: r for r in pq.read_table(a.pool).to_pylist()}
    protected = set(json.loads(a.splits_summary.read_text())['training_protected_library'])
    dirs = [LIBRARY / 'out/play'] + [r / 'out/play' for r in REFS]
    for s in a.screen:
        dirs += [s / 'out/probe', s / 'out/fill']
    dirs = [d for d in dirs if (d / 'results.jsonl').exists()]
    found = screened(dirs)
    rows, skipped = [], []
    for deck, recs in sorted(found.items()):
        if deck not in pool or deck in protected:
            skipped.append(dict(deck_id=deck, reason='not in pool' if deck not in pool else 'training-protected'))
            continue
        done = [x for x in recs if x[2]['status'] == 'completed']
        wins = sum(x[2]['fight']['won'] for x in done)
        secs = statistics.mean(x[2]['seconds'] for x in recs)
        row = dict(deck_id=deck, bucket=bucket(pool[deck]), hp=pool[deck]['start']['hp'],
                   max_hp=pool[deck]['start']['max_hp'], deck_size=len(pool[deck]['start']['deck']),
                   screen_n=len(recs), screen_completed=len(done), screen_wins=wins, screen_seconds=secs,
                   teacher_fight_ids=sorted(x[0] for x in done), teacher_runs=sorted({str(x[1]) for x in done}))
        if len(done) < 20 or len(recs) < 20:
            skipped.append(dict(row, reason='incomplete screen (software/cost scope or probe-only)'))
        elif secs > a.max_seconds:
            skipped.append(dict(row, reason='cost'))
        elif wins == 0:
            skipped.append(dict(row, reason='0/20 MCTS (not attempted tonight)'))
        elif a.check_encodable and not encodable(row, a.worker or WORKER):
            skipped.append(dict(row, reason='pv-encoder unsupported card'))
        else:
            rows.append(row)
    p_of = lambda r: r['screen_wins'] / 20
    key = lambda r: (abs(p_of(r) - .5), h(r['deck_id']))
    pilot = [r for r in rows if r['deck_id'] == A]
    if len(pilot) != 1:
        raise ValueError('Barricade A must be eligible')
    cand = sorted([r for r in rows if r['deck_id'] != A and 3 <= r['screen_wins'] <= 17
                   and r['screen_seconds'] <= a.pilot_seconds], key=key)
    for b in BUCKETS:
        if b == pilot[0]['bucket']:
            continue
        hit = [r for r in cand if r['bucket'] == b and r not in pilot]
        if hit and len(pilot) < a.pilot:
            pilot.append(hit[0])
    for r in cand:
        if len(pilot) >= a.pilot:
            break
        if r not in pilot:
            pilot.append(r)
    rest = [r for r in rows if r not in pilot]
    normal = [r for r in rest if r['screen_wins'] < 20]
    queues = {b: sorted([r for r in normal if r['bucket'] == b], key=key) for b in BUCKETS}
    ordered = []
    while any(queues.values()):
        for b in BUCKETS:
            if queues[b]:
                ordered.append(queues[b].pop(0))
    ordered += sorted([r for r in rest if r['screen_wins'] == 20], key=lambda r: h(r['deck_id']))
    decks = []
    for i, r in enumerate(pilot):
        decks.append(dict(r, wave=0, start=pool[r['deck_id']]))
    for i, r in enumerate(ordered):
        decks.append(dict(r, wave=1 + i // a.wave_size, start=pool[r['deck_id']]))
    # Filtered teacher source tables: only selected decks' completed screen fights, original schemas.
    keep = {f for d in decks for f in d['teacher_fight_ids']}
    for name in ('fights', 'search'):
        tables = []
        for d in dirs:
            t = pq.ParquetFile(d / f'{name}-0.parquet').read()
            tables.append(t.filter(pc.is_in(t['fight_id'], pa.array(sorted(keep)))))
        schema = tables[0].schema
        tables = [t.cast(schema) for t in tables]
        src = a.out / 'teacher-src'; src.mkdir(exist_ok=True)
        pq.write_table(pa.concat_tables(tables), src / f'{name}-0.parquet', compression='zstd')
    got = set(pq.ParquetFile(a.out / 'teacher-src/fights-0.parquet').read()['fight_id'].to_pylist())
    if got != keep:
        raise ValueError(f'teacher source mismatch: missing {len(keep - got)}, extra {len(got - keep)}')
    (a.out / 'decks.json').write_text(json.dumps(decks, indent=1))
    (a.out / 'skipped.json').write_text(json.dumps(skipped, indent=1))
    print(f'eligible {len(rows)}; pilot {len(pilot)}; later waves {len(ordered)}; skipped {len(skipped)}')
    for d in decks:
        print(f'wave {d["wave"]:2} {d["deck_id"][:8]} {d["bucket"]:10} HP {d["hp"]}/{d["max_hp"]} '
              f'MCTS {d["screen_wins"]:2}/20 {d["screen_seconds"]:5.1f}s')
    for s in skipped:
        print('skip', s['deck_id'][:8], s['reason'], s.get('screen_wins'), round(s.get('screen_seconds') or 0))


if __name__ == '__main__':
    main()
