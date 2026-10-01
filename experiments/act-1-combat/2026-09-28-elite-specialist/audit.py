#!/usr/bin/env python3
"""Inventory candidate elite fights and approximate opening-deck diversity. No training or replay.

Run: PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-28-elite-specialist/audit.py
"""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from sts_combat_rl.query import connect

HERE = Path(__file__).parent
BASE = "combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search"
SELF = "combat_v3/2026-09-27/ab-selfplay-gen1-rest"
# One first-decision row per fight; a card signature ignores card order/zone but includes upgrades.
db = connect()
rows = db.sql(f"""
    select run_id, run_seed, episode_id, source_episode_id, encounter, floor, starting_hp,
           won, cards
    from combat_v3
    where run_id in ('{BASE}', '{SELF}') and category = 'elite'
      and row_kind = 'decision' and decision_index = 0
""").fetchall()
by = defaultdict(list)
for run, seed, episode, source, enc, floor, hp, won, cards in rows:
    if run == SELF:
        split = "train: gen0 replay of bucket 4"
    else:
        split = {0: "old dev: excluded", 1: "old dev: excluded", 2: "final: bucket 2-3",
                 3: "final: bucket 2-3", 4: "train: bucket 4", 5: "validation: bucket 5",
                 6: "train: buckets 6-9", 7: "train: buckets 6-9",
                 8: "train: buckets 6-9", 9: "train: buckets 6-9"}[seed % 10]
    # Opening combat card multiset, not a true master-deck ID (generated/temporary cards can differ).
    signature = tuple(sorted((c['card_id'], int(c['numeric'][0])) for c in cards if c['zone'] < 4))
    by[(split, enc)].append((seed, episode, source, floor, hp, won, signature))

order = ["train: buckets 6-9", "train: bucket 4", "train: gen0 replay of bucket 4",
         "validation: bucket 5", "final: bucket 2-3", "old dev: excluded"]
lines = ["# Elite data inventory", "", "Counts refer to fights with an opening decision row. The baseline is",
         f"`{BASE}`; the additional self-play source is `{SELF}`.", "", "| Source / partition | Encounter | Fights | Run seeds | Distinct opening card multisets | Floors ≤10 / 11–14 / ≥15 | Starting HP median | Wins |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
for split in order:
    for enc in ('gremlin_nob', 'lagavulin', 'three_sentries'):
        v = by[(split, enc)]
        if not v:
            continue
        floors = [sum(flo <= 10 for _, _, _, flo, *_ in v), sum(11 <= flo <= 14 for _, _, _, flo, *_ in v), sum(flo >= 15 for _, _, _, flo, *_ in v)]
        hps = sorted(x[4] for x in v)
        lines.append(f"| {split} | {enc} | {len(v)} | {len(set(x[0] for x in v))} | {len(set(x[6] for x in v))} | {' / '.join(map(str, floors))} | {hps[len(hps)//2]} | {sum(x[5] for x in v)} |")
lines += ["", "## Diversity checks", ""]
train = {x[6] for (part, _), vs in by.items() if part.startswith('train:') for x in vs}
for split in ("validation: bucket 5", "final: bucket 2-3"):
    for enc in ('gremlin_nob', 'lagavulin', 'three_sentries'):
        v = by[(split, enc)]
        lines.append(f"- {split}, {enc}: {sum(x[6] not in train for x in v)}/{len(v)} opening card multisets not present in training (this is **not** a count of new master decks).")
replay = [x for (part, _), vs in by.items() if part == 'train: gen0 replay of bucket 4' for x in vs]
source_ids = {x[2] for x in replay}
source_train = {x[1] for (part, _), vs in by.items() if part == 'train: bucket 4' for x in vs}
heldout_ids = {x[1] for (part, _), vs in by.items() if part.startswith(('validation:', 'final:', 'old dev:')) for x in vs}
lines += ["", f"- Self-play: {len(replay)} replays from {len(source_ids)} distinct source fight IDs; {len(source_ids - source_train)} sources missing from bucket-4 baseline; {len(source_ids & heldout_ids)} source IDs in held-out partitions.",
          "- Opening card multiset is an approximate deck fingerprint: it drops order and zone and retains card ID/upgraded; it can include fight-generated cards. Exact deck diversity requires inspecting stored fight-start snapshots.", ""]
out = HERE / 'DATA_AUDIT.md'
out.write_text('\n'.join(lines) + '\n')
# Fixed selections before any candidate evaluation; never select using outcomes.
for split, file, n in (("validation: bucket 5", 'validation_fights.csv', 150),
                        ("final: bucket 2-3", 'final_fights.csv', 500)):
    with (HERE / file).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('source_episode_id', 'run_seed', 'encounter', 'floor', 'starting_hp'))
        for enc in ('gremlin_nob', 'lagavulin', 'three_sentries'):
            fights = sorted(by[(split, enc)], key=lambda x: hashlib.sha256(str(x[1]).encode()).digest())
            assert len(fights) >= n
            writer.writerows((episode, seed, enc, floor, hp) for seed, episode, _, floor, hp, *_ in fights[:n])
print(f'Wrote {out} ({len(rows)} fights), validation_fights.csv and final_fights.csv')
