"""One-off human-derived Champ benchmark. See README.md; run from the repository root."""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq

from apps.megacrit_dump.champ_starts import reconstruct, run_id_of, SIM
from apps.megacrit_dump.schema import load

ROOT = Path(__file__).resolve().parents[2]
MASK = (1 << 64) - 1
SCALING = {'demon_form', 'inflame', 'spot_weakness', 'limit_break'}
COUNTERS = {'happy_flower', 'incense_burner', 'ink_bottle', 'nunchaku', 'pen_nib', 'sundial'}
CHANGED = {'bloodletting', 'hemokinesis', 'rupture', 'dramatic_entrance', 'good_instincts',
           'sadistic_nature', 'swift_strike'}


def enum_ids(header, name):
    text = (SIM / header).read_text()
    body = re.search(r'enum class ' + name + r'[^\{]*\{(.*?)\}', text, re.S).group(1)
    body = re.sub(r'//[^\n]*', '', body)
    result, value = {}, -1
    for item in body.split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split('=')
        value = int(parts[1].strip(), 0) if len(parts) == 2 else value + 1
        result[parts[0].strip().lower()] = value
    return result


def rng_state(seed):
    # sts::Random::setSeed / libgdx RandomXS128, no draws yet.
    def murmur(x):
        x ^= x >> 33
        x = x * (-49064778989728563 & MASK) & MASK
        x ^= x >> 33
        x = x * (-4265267296055464877 & MASK) & MASK
        return (x ^ (x >> 33)) & MASK
    first = murmur(seed if seed else 1 << 63)
    return {'counter': 0, 'seed0': first, 'seed1': murmur(first)}


def convert(ev, row, seed):
    cards = enum_ids('Cards.h', 'CardId')
    relics = enum_ids('Relics.h', 'RelicId')
    rooms = enum_ids('Rooms.h', 'Room')
    encounters = enum_ids('MonsterEncounters.h', 'MonsterEncounter')
    if not row['exact']:
        raise ValueError('reconstruction:' + ','.join(sorted({i.split(':')[0] for i in row['issues']})))
    if 'runic_dome' in row['relics']:
        raise ValueError('runic_dome')
    if 'lizard_tail' in row['relics']:
        raise ValueError('unknown_lizard_tail_usage')
    if any(c['card'] in {'ritual_dagger', 'genetic_algorithm'} for c in row['deck']):
        raise ValueError('unknown_card_misc')
    if not row['deck'] or not (0 < row['hp'] <= row['max_hp'] <= 32767):
        raise ValueError('invalid_hp_or_deck')
    deck = []
    for c in row['deck']:
        n = c['upgrades']
        if n < 0 or (n > 1 and c['card'] != 'searing_blow'):
            raise ValueError('invalid_upgrade_count')
        deck.append({'id': cards[c['card']], 'upgraded': n > 0,
                     'misc': n if c['card'] == 'searing_blow' else 0})
    # Preserve acquisition order from the final record, except removed post-Champ relics.
    from apps.megacrit_dump.champ_starts import RELICS, CURSES
    order = [RELICS[r].lower() for r in ev['relics'] if r in RELICS and RELICS[r].lower() in row['relics']]
    order += [r for r in row['relics'] if r not in order]
    if len(order) != len(set(order)):
        raise ValueError('duplicate_relics')
    relic_rows = []
    for r in order:
        data = 0
        if r == 'du_vu_doll':
            data = sum(re.sub(r'\+\d+$', '', c) in CURSES for c in row['deck_raw'])
        elif r == 'girya':
            data = sum(c.get('key') == 'LIFT' and c.get('floor', 0) < row['F']
                       for c in ev.get('campfire_choices', []))
        relic_rows.append({'id': relics[r], 'data': data})
    path = ev.get('path_per_floor') or []
    symbol = path[row['F'] - 2] if len(path) >= row['F'] - 1 else None
    previous = {'R': 'rest', 'M': 'monster', 'E': 'elite', '?': 'event', '$': 'shop', 'T': 'treasure'}
    if symbol not in previous and 'ancient_tea_set' in order:
        raise ValueError('unknown_previous_room')
    return dict(seed=seed, ascension=20, act=2, floor=row['F'], encounter=encounters['champ'],
                cur_room=rooms['boss'], last_room=rooms[previous.get(symbol, 'none')],
                burning_elite_buff=-1, hp=row['hp'], max_hp=row['max_hp'], gold=0,
                misc_rng=rng_state(seed), potion_rng=rng_state(seed), potion_capacity=2,
                potions=[1, 1], relics=relic_rows, deck=deck, bottled=[-1, -1, -1])


def prepare(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    candidates, seen = [], set()
    versions, exclusions = collections.Counter(), []
    for source in a.runs:
        directory = source / 'out/runs' if (source / 'out/runs').is_dir() else source
        for file in sorted(directory.glob('part-*.parquet')):
            for batch in pq.ParquetFile(file).iter_batches(columns=['reached_champ', 'raw']):
                for record in batch.to_pylist():
                    if not record['reached_champ']:
                        continue
                    ev = json.loads(record['raw'])
                    pid = str(ev.get('play_id'))
                    if pid in seen:
                        continue
                    seen.add(pid)
                    row = reconstruct(ev)
                    versions[ev.get('build_version')] += 1
                    try:
                        if ev.get('character_chosen') != 'IRONCLAD' or ev.get('ascension_level') != 20:
                            raise ValueError('not_ironclad_a20')
                        if not str(ev.get('build_version', '')).startswith('2020-'):
                            raise ValueError('not_2020_build')
                        seed = int(ev['seed_played']) & MASK
                        start = convert(ev, row, seed)
                    except (ValueError, KeyError, TypeError) as error:
                        exclusions.append({'play_id': pid, 'human_won': row['champ_won'], 'reason': str(error)})
                        continue
                    candidates.append((ev, row, start, run_id_of(directory)))
    candidates.sort(key=lambda x: x[1]['play_id'])
    random.Random(a.sample_seed).shuffle(candidates)
    selected = candidates[:a.limit]
    rows = []
    for ev, row, start, source in selected:
        pid = row['play_id']
        fresh = int.from_bytes(hashlib.sha256(f'{a.sample_seed}:{pid}'.encode()).digest()[:8], 'little')
        if fresh == start['seed']:
            fresh ^= 1
        names = [c['card'] for c in row['deck']]
        for seed_kind, seed in [('human', start['seed']), ('fresh', fresh)]:
            s = dict(start, seed=seed, misc_rng=rng_state(seed), potion_rng=rng_state(seed))
            rows.append(dict(fight_id=f'human-champ:{pid}:{seed_kind}', deck_id=pid, source_run_id=source,
                             build_version=row['build_version'], human_won=row['champ_won'], seed_kind=seed_kind,
                             start=s, demon_form='demon_form' in names, scaling_count=sum(c in SCALING for c in names),
                             hp_band='low' if row['hp'] / row['max_hp'] < .5 else
                                     'mid' if row['hp'] / row['max_hp'] < .8 else 'high',
                             changed_cards=sorted(set(names) & CHANGED),
                             reset_counters=sorted(set(row['relics']) & COUNTERS)))
    schema = load('human_champ_bench_v1').STARTS
    pq.write_table(pa.Table.from_pylist(rows, schema), out / 'starts.parquet', compression='zstd')
    (out / 'skipped.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in exclusions))
    summary = dict(reached_champ=len(seen), eligible_decks=len(candidates), selected_decks=len(selected),
                   starts=len(rows), builds=dict(versions),
                   skipped=dict(collections.Counter(r['reason'] for r in exclusions)),
                   human_wins_all=sum(r['human_won'] for r in exclusions) + sum(r['champ_won'] for _, r, _, _ in candidates),
                   human_wins_selected=sum(r['champ_won'] for _, r, _, _ in selected), sample_seed=a.sample_seed)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def play_one(job):
    command, row, timeout = job
    begin = time.monotonic()
    try:
        proc = subprocess.run(command, input=json.dumps(row) + '\n', text=True, capture_output=True, timeout=timeout)
        if proc.returncode:
            raise RuntimeError(f'exit {proc.returncode}: {proc.stderr[-1500:]}')
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        result.update(fight_id=row['fight_id'], seconds=time.monotonic() - begin)
        return result
    except (subprocess.TimeoutExpired, RuntimeError, ValueError, IndexError) as error:
        return dict(fight_id=row['fight_id'], status='error', error=str(error), seconds=time.monotonic() - begin)


def play(a):
    if a.agent == 'pv' and not a.model:
        raise ValueError('--agent pv requires --model')
    command = [str(a.worker.resolve())] + (['teacher', str(a.sims)] if a.agent == 'teacher' else
                                          ['play', str(a.model.resolve()), str(a.sims)])
    if getattr(a, 'explore', False) or getattr(a, 'sample_turns', False):
        if a.agent != 'pv':
            raise ValueError('exploration flags require --agent pv')
        command += ['--explore'] * getattr(a, 'explore', False)
        command += ['--sample-turns'] * getattr(a, 'sample_turns', False)
    a.out.mkdir(parents=True, exist_ok=True)
    journal = a.out / 'results.jsonl'
    settings = dict(command=command, starts=str(a.starts.resolve()),
                    starts_sha=hashlib.sha256(a.starts.read_bytes()).hexdigest(),
                    worker_sha=hashlib.sha256(a.worker.read_bytes()).hexdigest())
    if a.model:
        settings['model_sha'] = hashlib.sha256(a.model.read_bytes()).hexdigest()
        external = a.model.with_name(a.model.name + '.data')
        if external.exists():
            settings['model_data_sha'] = hashlib.sha256(external.read_bytes()).hexdigest()
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            done[r['fight_id']] = r
        old = json.loads((a.out / 'settings.json').read_text())
        if any(settings.get(k) != v for k, v in old.items()):
            raise ValueError('cannot resume with different settings')
    (a.out / 'settings.json').write_text(json.dumps(settings, indent=2))
    rows = pq.read_table(a.starts).to_pylist()
    if a.limit:
        rows = rows[:a.limit]
    jobs = [(command, r, a.timeout) for r in rows if r['fight_id'] not in done]
    with ThreadPoolExecutor(a.workers) as pool, journal.open('a') as stream:
        for result in pool.map(play_one, jobs):
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            done[result['fight_id']] = result
            print(result['fight_id'], result['status'], round(result['seconds'], 1), flush=True)
    fights = [r['fight'] for r in done.values() if r['status'] == 'completed']
    pq.write_table(pa.Table.from_pylist(fights, load('combat_v4').FIGHTS), a.out / 'fights-0.parquet', compression='zstd')
    searches = [s for r in done.values() if r['status'] == 'completed' for s in r.get('search', [])]
    pq.write_table(pa.Table.from_pylist(searches, load('combat_v4').SEARCH), a.out / 'search-0.parquet', compression='zstd')
    summary = dict(attempted=len(done), completed=len(fights), wins=sum(r['won'] for r in fights),
                   statuses=dict(collections.Counter(r['status'] for r in done.values())),
                   decision_seconds=sum(s.get('seconds', 0) for r in done.values() for s in r.get('stats', [])))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def report(a):
    rows = pq.read_table(a.starts).to_pylist()
    results = []
    for directory in [a.teacher, a.pv]:
        results.append({r['fight_id']: r for r in map(json.loads, (directory / 'results.jsonl').read_text().splitlines())})
    decks = collections.defaultdict(list)
    for row in rows:
        decks[row['deck_id']].append(row)
    paired, failures = [], []
    for pid, group in decks.items():
        if len(group) != 2 or any(d.get(r['fight_id'], {}).get('status') != 'completed' for d in results for r in group):
            continue
        item = dict(group[0], human=float(group[0]['human_won']))
        item['teacher'] = sum(results[0][r['fight_id']]['fight']['won'] for r in group) / 2
        item['pv'] = sum(results[1][r['fight_id']]['fight']['won'] for r in group) / 2
        paired.append(item)
        if item['human'] and not item['teacher'] and not item['pv']:
            failures.extend(group)
    def estimate(values):
        n = len(values)
        if not n:
            return 'n/a'
        mean = sum(values) / n
        se = math.sqrt(sum((v - mean) ** 2 for v in values) / (n * (n - 1))) if n > 1 else None
        return f'{100 * mean:.1f}%' + (f' ± {100 * se:.1f} pts' if se is not None else ' (SE unavailable)')
    lines = ['# Human-derived Champ benchmark', '',
             f'{len(paired)} paired decks / {len(decks)} selected; two seeds per agent per deck.',
             'Rates are deck means ± 1 SE across decks (not 95% intervals).',
             'No potions; modern simulator; canonical deck order and fresh RNG streams. Unknown cyclic relic counters reset to 0.',
             'Exact deck reconstruction selects disproportionately for human losses; this is diagnostic, not a skill ranking.', '',
             '| group | decks | human | MCTS | PV | PV − MCTS |', '|---|---:|---:|---:|---:|---:|']
    source_summary = a.starts.parent / 'summary.json'
    if source_summary.exists():
        source = json.loads(source_summary.read_text())
        if 'human_wins_all' in source:
            lines[2:2] = [f'Source human wins: {source["human_wins_all"]}/{source["reached_champ"]}; '
                          f'retained human wins: {source["human_wins_selected"]}/{source["selected_decks"]}. '
                          'Source rate is context only, not the agent comparator.', '']
    lines.insert(5, 'Historical card balance changes are not corrected in v1 (official v2.2 notes: '
                    'https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/3856733252019104438).')
    groups = [('all', paired)]
    for field in ['demon_form', 'scaling_count', 'hp_band']:
        buckets = collections.defaultdict(list)
        for r in paired:
            key = min(r[field], 3) if field == 'scaling_count' else r[field]
            buckets[str(key)].append(r)
        groups.extend((f'{field}={k}', v) for k, v in sorted(buckets.items()))
    for label, group in groups:
        cells = [estimate([r[k] for r in group]) for k in ['human', 'teacher', 'pv']]
        cells.append(estimate([r['pv'] - r['teacher'] for r in group]))
        lines.append(f'| {label} | {len(group)} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Human wins that agents lost', '', '| group | human wins | MCTS lost both seeds | PV lost both seeds | both lost both |', '|---|---:|---:|---:|---:|']
    for label, group in groups:
        wins = [r for r in group if r['human']]
        lines.append(f'| {label} | {len(wins)} | {sum(not r["teacher"] for r in wins)} | '
                     f'{sum(not r["pv"] for r in wins)} | {sum(not r["teacher"] and not r["pv"] for r in wins)} |')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    pq.write_table(pa.Table.from_pylist(failures[:40], load('human_champ_bench_v1').STARTS),
                   a.out / 'viewer_starts.parquet', compression='zstd')
    (a.out / 'summary.json').write_text(json.dumps(dict(paired_decks=len(paired), selected_decks=len(decks),
                                                     human_won_both_lost=len(failures) // 2), indent=2))
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--runs', nargs='+', type=Path, required=True)
    prep.add_argument('--limit', type=int, default=500, help='maximum decks')
    prep.add_argument('--sample-seed', type=int, default=0)
    run = sub.add_parser('play')
    run.add_argument('--starts', type=Path, required=True)
    run.add_argument('--agent', choices=['teacher', 'pv'], required=True)
    run.add_argument('--model', type=Path)
    run.add_argument('--explore', action='store_true', help='PV root noise for training collection')
    run.add_argument('--sample-turns', action='store_true', help='sample early-turn PV moves from visits')
    run.add_argument('--sims', type=int, required=True)
    run.add_argument('--workers', type=int, default=8)
    run.add_argument('--timeout', type=int, default=600, help='seconds per fight; failures are recorded, not losses')
    run.add_argument('--limit', type=int, help='maximum fight starts, for smoke tests')
    run.add_argument('--worker', type=Path, default=ROOT / 'build/pv/agents/combat/pv/pv_worker')
    rep = sub.add_parser('report')
    rep.add_argument('--starts', type=Path, required=True)
    rep.add_argument('--teacher', type=Path, required=True)
    rep.add_argument('--pv', type=Path, required=True)
    for command in [prep, run, rep]:
        command.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    {'prepare': prepare, 'play': play, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
