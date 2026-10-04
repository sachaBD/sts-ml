#!/usr/bin/env python3
"""Fight starts for PV play: held-out benches and generated self-play decks, as combat_v4 `fights`-style rows
(fight_id, start [, won, final_hp of the recorded teacher play]). Parquet out; sources are combat_v4 fights tables.

  starts.py bench    --source GLOB --encounter 39 --seed-min 981000000000 --seed-max 982000000000 --out F.parquet
  starts.py generate --source GLOB --encounter 39 --n 4000 --seed0 991000000000 --exclude-seed-min 981000000000
                     --exclude-seed-max 982000000000 [--add-p 0.3 --remove-p 0.1 --hp-p 0.3] --rng 0 --out F.parquet

generate --bench-copies C --n (source_count*C) --add-p 0 --remove-p 0 --hp-p 0: every provided
held-out source of the encounter (no Dome), C times, unchanged except fresh seeds. No exclusion range.
Adds source_fight_id provenance; never use this evaluation-only dataset as training input.

bench: the recorded fights of that encounter whose run seed lies in [seed-min, seed-max), unchanged (the recorded
teacher result is kept for a paired check).
generate: decks from real recorded fight starts of Act >= 2 (deck, relics, potions, HP, max HP, gold, RNG counters)
moved into the boss room of `encounter` (act 2 floor 33 for Act 2 bosses) with a fresh battle seed (seed0 + i).
Augmentation (independent per start): add 1-2 scaling cards (50% upgraded) with add-p, remove all scaling cards
with remove-p, and redraw HP uniformly in [0.25, 1] x max HP with hp-p. Source run seeds in the exclude range
(the bench seeds) are never used, nor starts with Runic Dome (PV search rejects hidden intents).
"""
import argparse
import random
import re
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CARDS_H = ROOT.parent / 'sts_lightspeed/include/constants/Cards.h'
RUNIC_DOME = 57  # sts::RelicId; PV search does not support hidden intents yet
BOSS_ROOM, REST_ROOM = 6, 1  # sts::Room::BOSS / REST
ACT2_BOSS_FLOOR = 33
SCALING = ['DEMON_FORM', 'INFLAME', 'SPOT_WEAKNESS', 'LIMIT_BREAK', 'METALLICIZE', 'BARRICADE', 'FEEL_NO_PAIN',
           'JUGGERNAUT', 'RUPTURE', 'DARK_EMBRACE']


def card_ids():
    text = CARDS_H.read_text()
    body = text[text.index('enum class CardId'):]
    body = body[body.index('{') + 1:body.index('}')]
    names = [re.sub(r'\s*=.*', '', n).strip() for n in body.split(',')]
    ids = {n: i for i, n in enumerate(n for n in names if n)}
    if ids.get('INVALID') != 0:
        raise ValueError('unexpected CardId enum layout')
    return ids


def start_type():
    import importlib.util
    spec = importlib.util.spec_from_file_location('combat_v4_schema', ROOT / 'runs/schema=combat_v4/schema.py')
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    return schema.START


def db():
    c = duckdb.connect()
    c.execute("set memory_limit='2GB'; set threads=4")
    return c


def bench(a):
    rel = db().sql(f"""select fight_id, start, won, final_hp from read_parquet('{a.source}')
        where start.encounter = {a.encounter} and start.seed >= {a.seed_min} and start.seed < {a.seed_max}
        order by fight_id""")
    rel.write_parquet(str(a.out), compression='zstd')
    print(f'{len(rel)} bench starts -> {a.out}')


def generate_bench(a):
    """Fresh seeds for every held-out start, with its complete loadout unchanged."""
    if a.bench_copies < 1 or any((a.add_p, a.remove_p, a.hp_p)):
        raise ValueError('--bench-copies requires copies >= 1 and --add-p 0 --remove-p 0 --hp-p 0')
    if a.exclude_seed_min is not None or a.exclude_seed_max is not None:
        raise ValueError('--bench-copies uses held-out sources: do not pass exclusion ranges')
    pool = db().sql(f"""select fight_id, start from read_parquet('{a.source}')
        where start.encounter = {a.encounter} and start.act >= 2
        and not list_contains(list_transform(start.relics, r -> r.id), {RUNIC_DOME})
        order by fight_id""").fetchall()
    if a.n != len(pool) * a.bench_copies:
        raise ValueError(f'--bench-copies needs n = {len(pool)} * {a.bench_copies}, not {a.n}')
    rows = []
    for source_id, source in pool:
        for _ in range(a.bench_copies):
            seed = a.seed0 + len(rows)
            rows.append({'fight_id': f'gen:{seed}', 'start': dict(source, seed=seed),
                         'augment': '', 'source_fight_id': source_id})
    schema = pa.schema([('fight_id', pa.string()), ('start', start_type()), ('augment', pa.string()),
                        ('source_fight_id', pa.string())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), a.out, compression='zstd')
    print(f'{len(pool)} held-out sources × {a.bench_copies} = {len(rows)} fresh starts -> {a.out}')


def generate(a):
    if a.bench_copies is not None:
        return generate_bench(a)
    if a.exclude_seed_min is None or a.exclude_seed_max is None:
        raise ValueError('normal generate requires --exclude-seed-min and --exclude-seed-max')
    rng = random.Random(a.rng)
    ids = card_ids()
    scaling = {ids[n] for n in SCALING}
    pool = db().sql(f"""select start from read_parquet('{a.source}')
        where start.act >= 2 and not list_contains(list_transform(start.relics, r -> r.id), {RUNIC_DOME}) and not (start.seed >= {a.exclude_seed_min} and start.seed < {a.exclude_seed_max})
        order by fight_id""").fetchall()
    if len(pool) < a.n:
        raise ValueError(f'only {len(pool)} source starts for {a.n} requested')
    rows = []
    for i, (src,) in enumerate(rng.sample(pool, a.n)):
        s = dict(src)
        deck = [dict(c) for c in s['deck']]
        bottled = list(s['bottled'])
        aug = []
        if rng.random() < a.remove_p:
            keep = [j for j, c in enumerate(deck) if c['id'] not in scaling]
            if len(keep) < len(deck):
                remap = {old: new for new, old in enumerate(keep)}
                bottled = [remap.get(b, -1) if b >= 0 else -1 for b in bottled]
                deck = [deck[j] for j in keep]
                aug.append('remove')
        if rng.random() < a.add_p:
            for _ in range(rng.choice((1, 2))):
                deck.append({'id': ids[rng.choice(SCALING)], 'upgraded': rng.random() < 0.5, 'misc': 0})
            aug.append('add')
        if rng.random() < a.hp_p:
            s['hp'] = max(1, round(s['max_hp'] * rng.uniform(0.25, 1.0)))
            aug.append('hp')
        s.update(seed=a.seed0 + i, act=2, floor=ACT2_BOSS_FLOOR, encounter=a.encounter, cur_room=BOSS_ROOM,
                 last_room=REST_ROOM, burning_elite_buff=-1, deck=deck, bottled=bottled)
        rows.append({'fight_id': f'gen:{a.seed0 + i}', 'start': s, 'augment': ','.join(aug)})
    table = pa.Table.from_pylist(rows, schema=pa.schema([('fight_id', pa.string()), ('start', start_type()),
                                                          ('augment', pa.string())]))
    pq.write_table(table, a.out, compression='zstd')
    print(f'{len(rows)} generated starts -> {a.out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('bench')
    g = sub.add_parser('generate')
    for p in (b, g):
        p.add_argument('--source', required=True, help='glob of combat_v4 fights parquet files')
        p.add_argument('--encounter', type=int, required=True)
        p.add_argument('--out', type=Path, required=True)
    b.add_argument('--seed-min', type=int, required=True)
    b.add_argument('--seed-max', type=int, required=True)
    g.add_argument('--n', type=int, required=True)
    g.add_argument('--seed0', type=int, required=True)
    g.add_argument('--exclude-seed-min', type=int)
    g.add_argument('--exclude-seed-max', type=int)
    g.add_argument('--bench-copies', type=int, help='repeat every provided held-out source, no augmentation/exclusion')
    g.add_argument('--add-p', type=float, default=0.3)
    g.add_argument('--remove-p', type=float, default=0.1)
    g.add_argument('--hp-p', type=float, default=0.3)
    g.add_argument('--rng', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'generate' and a.bench_copies is None and (a.exclude_seed_min is None or a.exclude_seed_max is None):
        ap.error('normal generate requires --exclude-seed-min and --exclude-seed-max')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bench(a) if a.cmd == 'bench' else generate(a)


if __name__ == '__main__':
    main()
