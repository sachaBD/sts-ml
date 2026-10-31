"""Freeze train/dev/final compositions and inputs. No agent outcome is used to choose a deck."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import tomllib

import pyarrow as pa
import pyarrow.parquet as pq

from environments.combat.starts import fights
from apps.gumbel_nn_combat_expert_iteration.train import load_bases

AUDIT = runpy.run_path('experiments/2026-10-28-champ-generalisation/inventory.py')
identity, compatible = AUDIT['identities'], AUDIT['compatible']
HERE = Path(__file__).parent
OLD = Path('runs/schema=gumbel_nn_combat_v1/date=2026-10-28/id=gumbel-nn-champ-fifty-20261028/out')
POOL = Path('runs/schema=combat_v4/date=2026-10-05/id=human-deck-corpus/splits/pool.parquet')
SOURCES = [POOL, Path('runs/schema=combat_v4/date=2026-10-05/id=human-combat-r01/out/source/starts.parquet'),
           Path('runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet')]
NAMESPACE = 'gumbel-nn-champ-1000-20261028-night'


def stable(start):
    return {k: v for k, v in start.items() if k not in {'seed', 'misc_rng', 'potion_rng'}}


def rank(row, split):
    return hashlib.sha256((NAMESPACE + ':' + split + ':' + identity(row['start'])[1]).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'split.json').exists(): raise FileExistsError('refusing to replace a frozen split')
    oldconfig = tomllib.loads((OLD / 'config.toml').read_text())
    oldbases = load_bases(oldconfig['deck'])
    old_by_comp = {identity(row['start'])[1]: (name, row) for name, row in oldbases.items()}
    assert len(old_by_comp) == 50
    source_rows = {}; pool = {}
    for source in SOURCES:
        for row in pq.ParquetFile(source).read().to_pylist():
            if row['seed_kind'] != 'human' or not compatible(row['start']): continue
            comp = identity(row['start'])[1]
            source_rows.setdefault(comp, row)
            if source == POOL: pool[comp] = row
    train = [row for _, row in old_by_comp.values()]
    train += sorted([r for comp, r in pool.items() if comp not in old_by_comp], key=lambda r: rank(r, 'train'))[:950]
    assert len(train) == 1000
    train_comps = {identity(r['start'])[1] for r in train}; train_ids = {r['deck_id'] for r in train}
    # Protect the entire historical training pool, including the few pool decks
    # not selected tonight. This is stricter than just the current NN lineage.
    pool_ids = {r['deck_id'] for r in pool.values()}
    held = [r for comp, r in source_rows.items() if comp not in pool and comp not in train_comps
            and r['deck_id'] not in train_ids | pool_ids]
    held = sorted(held, key=lambda r: rank(r, 'heldout'))
    assert len(held) >= 400
    dev, final = held[:200], held[200:400]
    monitor_new = sorted([r for r in train if identity(r['start'])[1] not in old_by_comp], key=lambda r: rank(r, 'monitor'))[:50]
    panel = [*[(row, 'legacy50') for _, row in old_by_comp.values()],
             *[(row, 'train_monitor') for row in monitor_new], *[(row, 'unseen_dev') for row in dev]]
    def name(row):
        return old_by_comp.get(identity(row['start'])[1], (f'deck-{row["deck_id"][:12]}', None))[0]
    names = [name(r) for r in train + dev + final]
    assert len(names) == len(set(names)) == 1400
    # Canonical historical policies only; no pooling different PUCT checkpoints/search modes.
    teacher_path = Path('runs/schema=combat_v4/date=2026-10-05/id=human-champ-v1-teacher/out/fights-0.parquet')
    puct_path = Path('runs/schema=combat_v4/date=2026-10-05/id=human-champ-v1-pv/out/fights-0.parquet')
    def historical(path):
        return {json.dumps(r['start'], sort_keys=True): r for r in pq.ParquetFile(path).read(columns=['start','won','fight_id','agent']).to_pylist()}
    teacher, puct = historical(teacher_path), historical(puct_path)
    cached = {}
    for key in teacher.keys() & puct.keys():
        t, q = teacher[key], puct[key]
        if 'sims=20000' not in t['agent'] or 'sims=2000 ' not in q['agent'] or 'explore=0 ' not in q['agent']: continue
        cached.setdefault(identity(t['start'])[1], []).append((t, q))
    baselines = dict(mcts20k_historical={}, puct_D5_historical={})
    def validation(rows, split, count, baseline):
        output = []
        for row in rows:
            deck = name(row); base = dict(fight_id=deck, start=row['start'])
            available = [(t,q) for t,q in cached.get(identity(row['start'])[1], []) if stable(t['start']) == stable(row['start'])]
            available = sorted(available, key=lambda pair: pair[0]['start']['seed'])[:2]
            prefix = f'{NAMESPACE}:{deck}:{split}'
            for i, (t, q) in enumerate(available):
                fid = f'{prefix}:{i}'
                output.append(dict(fight_id=fid, start=t['start']))
                baseline['mcts20k_historical'][fid] = t['won']; baseline['puct_D5_historical'][fid] = q['won']
            output += fights(base, prefix, count)[len(available):]
        assert len({r['fight_id'] for r in output}) == len(output)
        return output
    primary = validation(dev, 'dev', 10, baselines)
    insample = validation([row for row, role in panel if role != 'unseen_dev'], 'monitor', 10, baselines)
    final_baselines = dict(mcts20k_historical={}, puct_D5_historical={})
    reserved = validation(final, 'final', 20, final_baselines)
    assert len(primary)==2000 and len(insample)==1000 and len(reserved)==4000
    dev_comps={identity(r['start'])[1] for r in dev};final_comps={identity(r['start'])[1] for r in final}
    assert not train_comps & dev_comps and not train_comps & final_comps and not dev_comps & final_comps
    assert not set(old_by_comp) & (dev_comps | final_comps)
    assert not set(pool) & (dev_comps | final_comps)
    schema=pq.ParquetFile(POOL).schema_arrow
    pq.write_table(pa.Table.from_pylist(train,schema=schema),out/'train-starts.parquet')
    for filename, records in [('dev.jsonl',primary),('monitor.jsonl',insample),('final.jsonl',reserved)]:
        (out/filename).write_text(''.join(json.dumps(r)+'\n' for r in records))
    write_json(out/'baselines.json',baselines);write_json(out/'final-baselines.json',final_baselines)
    cohorts={name(row):role for row,role in panel};write_json(out/'cohorts.json',cohorts)
    split = [dict(name=name(row), deck_id=row['deck_id'], composition=identity(row['start'])[1], role=role)
             for role, rows in [('train', train), ('dev', dev), ('final', final)] for row in rows]
    write_json(out/'split.json',split)
    write_json(out/'baseline-provenance.json',dict(mcts20k=str(teacher_path),puct_D5=str(puct_path),
        warning='Historical simulator/model; same full start and seed paired. Not current-rules controlled baselines.'))
    text=f'''# Thousand-deck unseen-composition generalisation. Inputs frozen before gameplay.
[run]
id = "{NAMESPACE}"
worker = "build/nn/agents/combat/nn/nn_play"
warm_start_run = "gumbel_nn_combat_v1/2026-10-28/gumbel-nn-champ-fifty-20261028"
recovery_retries = 2
games_per_deck = 5
seed_namespace = "{NAMESPACE}"

[loop]
balanced_replay = true
iterations = 0
validation_share = 0.20
test_every = 0
prune_replay = true
min_free_gib = 10
steps = 500
window = 10
max_train_rows = 0
batch = 2048
lr = 0.001
sims = 256
rollout_mix = 0.5
workers = 9
parallel = 224
scheduler = "local"
groups = 2
gpu_batch = 512
gpu_tokens = 64
gpu_bank = "256,32"
gpu_wait_us = 100
timeout_ms = 60000
shutdown_timeout_ms = 5000
dim = 64
layers = 2
heads = 2
device = "cuda"

[evaluation]
starts = "{out / 'dev.jsonl'}"
train_starts = "{out / 'monitor.jsonl'}"
cohorts = "{out / 'cohorts.json'}"
baselines = "{out / 'baselines.json'}"
final_starts = "{out / 'final.jsonl'}"
at = [0]
'''
    for row in train:
        s=row['start']; d=dict(name=name(row),starts=str(out/'train-starts.parquet'),deck_id=row['deck_id'],hp=s['hp'],max_hp=s['max_hp'])
        text+='\n[[deck]]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in d.items())+'\n'
    (HERE/'champ_1000.toml').write_text(text)
    report=dict(train=1000,dev=200,final=200,legacy=50,monitor=100,
        primary_trials=len(primary),monitor_trials=len(insample),final_trials=len(reserved),
        matched_historical_primary=sum(r['fight_id'] in baselines['mcts20k_historical'] for r in primary),
        split_sha256=hashlib.sha256((out/'split.json').read_bytes()).hexdigest(), input_dir=str(out))
    write_json(out/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
