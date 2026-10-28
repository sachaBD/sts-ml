"""Deterministic fifty-deck recipe; no gameplay and no selection using agent win rates."""
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import tomllib

import pyarrow.parquet as pq

from apps.human_champ.bench import enum_ids
from apps.combat_expert_iteration.expert_iteration import load_base

HERE = Path(__file__).parent
SOURCE = 'runs/schema=combat_v4/date=2026-10-05/id=human-deck-corpus/splits/pool.parquet'
CARDS = {v: k for k, v in enum_ids('Cards.h', 'CardId').items()}
RELICS = enum_ids('Relics.h', 'RelicId')
COMMON = {'strike_red', 'defend_red', 'ascenders_bane', 'bash'}
# Mirror environment.cpp's deliberate encoder boundary, not the wider simulator pool.
headers = Path('../lightspeed/include/constants')
colors = re.findall(r'CardColor::(\w+)', re.search(r'cardColors\[\].*?\{(.*?)\}',
                    (headers / 'Cards.h').read_text(), re.S).group(1))
colorless = re.findall(r'CardId::\s*(\w+)', re.search(r'baseColorlessPool.*?\{(.*?)\}',
                       (headers / 'CardPools.h').read_text(), re.S).group(1))
extras = {n.lower() for n in colorless} | {'apparition', 'bite', 'jax', 'ritual_dagger', 'burn', 'dazed', 'slimed', 'void', 'wound'}
SUPPORTED = {i for i, name in CARDS.items() if colors[i] in {'RED', 'CURSE'} or name in extras}


def names(row):
    return {CARDS[c['id']] for c in row['start']['deck']}


def family(row):
    return tuple(sorted(Counter((c['id'], c['upgraded'], c['misc']) for c in row['start']['deck']).items()))


def bucket(row):
    cards = names(row)
    if cards & {'barricade', 'entrench'}: return 'block'
    if 'demon_form' in cards: return 'demon_form'
    if cards & {'corruption', 'dark_embrace', 'feel_no_pain'}: return 'exhaust'
    if cards & {'limit_break', 'inflame', 'spot_weakness', 'jax', 'heavy_blade', 'flex'}: return 'strength'
    return 'mixed'


def distance(a, b):
    x, y = names(a) - COMMON, names(b) - COMMON
    c = 1 - len(x & y) / max(1, len(x | y))
    x, y = ({r['id'] for r in s['start']['relics']} for s in (a, b))
    return .8 * c + .2 * (1 - len(x & y) / max(1, len(x | y)))


def main():
    original = Path('apps/gumbel_nn_combat_expert_iteration/config/champ_ten.toml')
    config = tomllib.loads(original.read_text())
    chosen = [load_base(d)[0] for d in config['deck']]
    used = {family(r) for r in chosen}
    rows = pq.read_table(SOURCE).to_pylist()
    candidates = [r for r in rows if r['seed_kind'] == 'human' and r['start']['ascension'] == 20
                  and r['start']['encounter'] == 39 and r['start']['potions'] == [1, 1]
                  and 8 <= len(r['start']['deck']) <= 40
                  and all(c['id'] in SUPPORTED for c in r['start']['deck'])
                  and not {x['id'] for x in r['start']['relics']} &
                      {RELICS['runic_dome'], RELICS['gambling_chip'], RELICS['prismatic_shard']}]
    added = []
    for group in ['block', 'demon_form', 'exhaust', 'strength', 'mixed']:
        for band in ['mid', 'high', 'low', 'mid', 'high', 'low', 'mid', 'high']:
            available = [r for r in candidates if bucket(r) == group and r['hp_band'] == band and family(r) not in used]
            row = max(available, key=lambda r: (min(distance(r, s) for s in chosen),
                      hashlib.sha256(r['deck_id'].encode()).hexdigest()))
            chosen.append(row); used.add(family(row))
            d = dict(name=f'{group}-{row["deck_id"][:8]}', starts=SOURCE, deck_id=row['deck_id'],
                     hp=row['start']['hp'], max_hp=row['start']['max_hp'])
            added.append(d)
    assert len(chosen) == len(used) == 50
    text = original.read_text().replace('gumbel-nn-champ-ten-v1', 'gumbel-nn-champ-fifty-20261028')
    text = text.replace('games_per_deck = 500 # 5,000 total/update; replacement reserve beyond 9 x 224 active slots',
                        'games_per_deck = 100 # 5,000 total/update across fifty decks')
    text = text.replace('test_every = 3', 'test_every = 0\nvalidation_share = 0.20 # measured validation wall-time budget, checked at update boundaries')
    text = text.replace('[loop]\n', '[loop]\nbalanced_replay = true\n')
    text = text.replace('fights_per_deck = 50', 'fights_per_deck = 50 # stored historical reference for original ten only\ngenerated_per_deck = 100')
    text = text.replace('[run]\n', '[run]\nwarm_start_run = "gumbel_nn_combat_v1/2026-10-27/gumbel-nn-champ-ten-20261027-205535"\n')
    text = text.replace('# Example recipe, not an approved experiment budget. Counts below are PER DECK.',
                        '# Authorized fifty-deck expansion. Run until stopped; original ten retained verbatim.')
    for d in added:
        text += '\n[[deck]]\n' + '\n'.join(f'{k} = {json.dumps(v)}' for k, v in d.items()) + '\n'
    (HERE / 'champ_fifty.toml').write_text(text)
    manifest = [dict(name=d['name'], cohort='original' if i < 10 else 'added', deck_id=r['deck_id'],
                    bucket=bucket(r), hp=r['start']['hp'], max_hp=r['start']['max_hp'], cards=len(r['start']['deck']),
                    card_names=sorted(names(r)), hp_band=r['hp_band'])
                for i, (d, r) in enumerate(zip(config['deck'] + added, chosen))]
    (HERE / 'decks.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# Fifty-deck selection', '',
             'Original ten unchanged. Forty added by deterministic farthest-first card/relic diversity;',
             'eight per mechanic bucket (3 mid / 3 high / 2 low HP), exact deck families deduplicated.',
             'No agent win-rate filtering. Source is the existing reconstructed/supported A20 Champ pool.',
             'Encoder-incompatible cards and Prismatic Shard excluded (cross-class capability).',
             'Runic Dome excluded (unsupported intents); Gambling Chip excluded in this expansion',
             'to avoid exponential initial selection branches. Added deck sizes limited to 8–40 cards.', '',
             '| Name | Cohort | Bucket | HP | Cards |', '|---|---|---|---:|---:|']
    lines += [f'| {r["name"]} | {r["cohort"]} | {r["bucket"]} | {r["hp"]}/{r["max_hp"]} | {r["cards"]} |' for r in manifest]
    (HERE / 'DECKS.md').write_text('\n'.join(lines) + '\n')
    print('Selected fifty unique deck families; wrote champ_fifty.toml, decks.json and DECKS.md')


if __name__ == '__main__':
    main()
