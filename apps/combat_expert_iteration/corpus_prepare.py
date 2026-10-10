"""Freeze the local pilot's starts and provenance; support-only, never outcome-ranked.

The prior family holdout is preserved, then reconciled with later recorded deck configs and
corpus admissions. Exact current-checkpoint ancestry is checked against its recorded starts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tomllib

import pyarrow.parquet as pq

from apps.common.app import sha256
from apps.human_champ.bench import SIM, enum_ids
from .corpus_io import atomic_json, freeze_json, identity
from .corpus_manifest import parse_manifest, validate_production_freeze, TEN_DECK_LOCATOR
from .expert_iteration import load_base

BUCKETS = ('block', 'demon_form', 'exhaust', 'strength', 'mixed')
LEGACY = Path(TEN_DECK_LOCATOR)
SOURCE = Path('runs/schema=combat_v4/date=2026-10-05/id=human-combat-r01/out/source/starts.parquet')
BENCH = Path('runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet')
HELD = Path('runs/schema=combat_v4/date=2026-10-05/id=human-deck-corpus/splits/heldout-families.json')


def family(row):
    # Deliberately stricter than the historical card+relic fingerprint: relic variants stay together.
    return identity(sorted((c['id'], c['upgraded'], c['misc']) for c in row['start']['deck']))


def old_family(row):
    s = row['start']
    obj = dict(deck=sorted((c['id'], c['upgraded'], c['misc']) for c in s['deck']),
               relics=sorted(r['id'] for r in s['relics']), ascension=s['ascension'], encounter=s['encounter'])
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def bucket(row, cards):
    names = {cards[c['id']] for c in row['start']['deck']}
    if names & {'barricade', 'entrench'}: return 'block'
    if 'demon_form' in names: return 'demon_form'
    if names & {'corruption', 'dark_embrace', 'feel_no_pain'}: return 'exhaust'
    if names & {'limit_break', 'inflame', 'spot_weakness', 'jax', 'heavy_blade', 'flex'}: return 'strength'
    return 'mixed'


def supported_ids():
    """Mirror declared encoder capability; subsequent smoke verifies the frozen binary too."""
    header = (SIM / 'Cards.h').read_text()
    colors = re.findall(r'CardColor::(\w+)', re.search(r'cardColors\[\].*?\{(.*?)\}', header, re.S)[1])
    types = re.findall(r'CardType::(\w+)', re.search(r'cardTypes\[\].*?\{(.*?)\}', header, re.S)[1])
    ids = enum_ids('Cards.h', 'CardId')
    pool = re.search(r'baseColorlessPool.*?\{(.*?)\}', (SIM / 'CardPools.h').read_text(), re.S)[1]
    names = re.findall(r'CardId::\s*(\w+)', pool) + ['APPARITION', 'BITE', 'JAX', 'RITUAL_DAGGER']
    return {i for i, (color, kind) in enumerate(zip(colors, types))
            if color in {'RED', 'CURSE'} or kind == 'STATUS'} | {ids[n.lower()] for n in names}


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'prepared.json').exists():
        return json.loads((out / 'prepared.json').read_text())
    cfg = tomllib.loads((LEGACY / 'config.toml').read_text())
    originals = {d['name']: load_base(d)[0] for d in cfg['deck']}
    original_families = {family(r) for r in originals.values()}
    if len(original_families) != 10:
        raise ValueError('original ten contain duplicate card-only families')
    cards = {v: k for k, v in enum_ids('Cards.h', 'CardId').items()}
    supported = supported_ids()
    rows = [r for p in (SOURCE, BENCH) for r in pq.ParquetFile(p).read().to_pylist() if r['seed_kind'] == 'human']
    held = set(json.loads(HELD.read_text()))
    # A card-only group is held out if ANY prior variant was held out.
    held_cards = {family(r) for r in rows if old_family(r) in held}
    known_ids = {r['deck_id'] for r in originals.values()}
    audit_files = {str(p): sha256(p) for p in (SOURCE, BENCH, HELD, LEGACY / 'config.toml')}
    for path in sorted(Path('runs/schema=combat_v4').glob('date=*/id=*/out/config.toml')):
        config = tomllib.loads(path.read_text())
        for d in config.get('deck', []):
            if 'deck_id' in d:
                known_ids.add(d['deck_id'])
        audit_files[str(path)] = sha256(path)
    for path in sorted(Path('runs/schema=combat_v4').glob('date=*/id=hdc-*/out/state.json')):
        known_ids.update(json.loads(path.read_text()).get('decks', {}))
        audit_files[str(path)] = sha256(path)
    # Recorded explicit historical training starts provide a conservative exposure exclusion.
    exposed_cards = set(original_families)
    for path in sorted(Path('runs/schema=combat_v4').glob('date=*/id=*/out/train.parquet')):
        if 'start' not in pq.ParquetFile(path).schema_arrow.names:
            continue
        for r in pq.ParquetFile(path).read(columns=['start']).to_pylist():
            exposed_cards.add(family(r))
        audit_files[str(path)] = sha256(path)
    exposed_cards.update(family(r) for r in rows if r['deck_id'] in known_ids)
    groups, skipped = {}, []
    for r in sorted(rows, key=lambda r: r['deck_id']):
        fid = family(r)
        unsupported = sorted({c['id'] for c in r['start']['deck']} - supported)
        if unsupported:
            skipped.append(dict(deck_id=r['deck_id'], reason='unsupported cards', cards=[cards[i] for i in unsupported]))
            continue
        if r['start']['ascension'] != 20 or r['start']['encounter'] != 39 or r['start']['potions'] != [1, 1]:
            skipped.append(dict(deck_id=r['deck_id'], reason='start contract'))
            continue
        if fid in original_families:
            continue
        groups.setdefault(fid, r)
    pools = {'pilot': {b: [] for b in BUCKETS}, 'held': {b: [] for b in BUCKETS}}
    for fid, row in groups.items():
        if fid in held_cards:
            if fid in exposed_cards:
                continue
            side = 'held'
        else:
            # New to incumbent and prior focused corpora; old broad models are not ancestors.
            if row['deck_id'] in known_ids:
                continue
            side = 'pilot'
        pools[side][bucket(row, cards)].append((fid, row))
    available = {s: {b: len(v) for b, v in buckets.items()} for s, buckets in pools.items()}
    atomic_json(out / 'inventory.json', dict(available=available, support_exclusions=skipped,
                                            audit_files=audit_files, known_deck_ids=sorted(known_ids)))
    selected = []
    for b in BUCKETS:
        train = sorted(pools['pilot'][b], key=lambda x: identity(['champ-corpus-pilot-v1', 'train', x[0]]))
        test = sorted(pools['held'][b], key=lambda x: identity(['champ-corpus-pilot-v1', 'held', x[0]]))
        if len(train) < 8 or len(test) < 24:
            raise ValueError(f'insufficient support-only inventory for {b}: {available}; review quotas before outcomes')
        for i, (fid, row) in enumerate(train[:8]):
            selected.append((fid, row, 'pilot', i // 2 + 1, b))
        for i, (fid, row) in enumerate(test[:24]):
            selected.append((fid, row, 'development' if i < 4 else 'final', None, b))
    selected += [(family(row), row, 'original', None, bucket(row, cards)) for row in originals.values()]
    bases, families = {}, []
    for fid, row, role, wave, b in selected:
        families.append(dict(family_id=fid, role=role, wave=wave, provenance=[f'cards:{fid}', f'play:{row["deck_id"]}']))
        bases[fid] = dict(row=row, bucket=b)
    legacy_shards, exposure, hashes = [], [], {}
    for k in range(7):
        path = LEGACY / ('bootstrap' if k == 0 else f'update{k:03d}') / 'play/rows/rows.parquet'
        digest = sha256(path)
        hashes[str(path)] = digest
        fights = {}
        for fid in sorted(set(pq.ParquetFile(path).read(columns=['fight_id'])['fight_id'].to_pylist())):
            name = fid.split(':')[1]
            if name not in originals:
                raise ValueError(f'unknown legacy deck alias: {name}')
            fights[fid] = dict(family_id=family(originals[name]), source='teacher' if k == 0 else 'learner', round=0)
        legacy_shards.append(dict(path=str(path.resolve()), sha256=digest, fights=fights))
        for family_id in sorted({m['family_id'] for m in fights.values()}):
            exposure.append(dict(source_id=f'{k}:{family_id}', family_id=family_id,
                                 category='bootstrap' if k == 0 else f'learner_update{k:03d}', used_for_training=True))
    worker = LEGACY / 'pv_worker'
    model = LEGACY / 'update006/model'
    for p in (worker, model / 'model.pt', model / 'model.onnx'):
        hashes[str(p)] = sha256(p)
    manifest = dict(schema='champ-corpus-pilot-v1', legacy_run=dict(run_id='champ-ten-rollout-v1',
                    locator=str(LEGACY), artifact_sha256=hashes), families=families,
                    historical_exposure=exposure, replay_source_ids=[e['source_id'] for e in exposure])
    validate_production_freeze(parse_manifest(manifest))
    freeze_json(out / 'families.json', manifest)
    freeze_json(out / 'bases.json', bases)
    freeze_json(out / 'legacy-shards.json', legacy_shards)
    prepared = dict(worker=str(worker.resolve()), model=str(model.resolve()), manifest_sha256=identity(manifest),
                    families=Counter(f['role'] for f in families), audit_scope='prior holdout + later configs/admissions + explicit train starts; current checkpoint ancestry exact',
                    warning='Historical broad training may have seen pilot training families, but is not checkpoint ancestry. Source dump ID is not a human run ID; grouping uses card family and play UUID.')
    freeze_json(out / 'prepared.json', prepared)
    return prepared


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), indent=2))
