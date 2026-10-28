"""Read-only inventory: deck compositions, launch compatibility and historical baseline coverage."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import runpy

import pyarrow.parquet as pq

OUT = Path('scratch/champ-generalisation-audit')
SCOPE = runpy.run_path('experiments/2026-10-28-champ-fifty/prepare.py')
SUPPORTED, RELICS = SCOPE['SUPPORTED'], SCOPE['RELICS']


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identities(start):
    exact = sorted(Counter((c['id'], c['upgraded'], c['misc']) for c in start['deck']).items())
    composition = sorted(Counter(c['id'] for c in start['deck']).items())
    stable = {k: v for k, v in start.items() if k not in {'seed', 'misc_rng', 'potion_rng'}}
    return key(exact), key(composition), key(stable)


def compatible(start):
    return (start['ascension'] == 20 and start['encounter'] == 39 and start['potions'] == [1, 1]
            and all(c['id'] in SUPPORTED for c in start['deck'])
            and not {r['id'] for r in start['relics']} & {RELICS['runic_dome'], RELICS['prismatic_shard']})


def classify(agent):
    if re.search(r'oracle=(1|true)', agent) or any(flag in agent for flag in ('root_halving', 'turn_search', 'real_turn', 'turn_targets', 'pimc=')):
        return None
    sims = re.search(r'\bsims=(\d+)', agent)
    if agent.startswith('mcts ') and sims and int(sims[1]) == 20000:
        return 'mcts20k'
    if agent.startswith('pv ') and sims and int(sims[1]) == 2000 and re.search(r'\bexplore=0(?:\s|$)', agent) and 'policy_only' not in agent:
        return 'puct'
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    families = {}
    sources = [
        ('pool', 'runs/schema=combat_v4/date=2026-10-05/id=human-deck-corpus/splits/pool.parquet'),
        ('r01', 'runs/schema=combat_v4/date=2026-10-05/id=human-combat-r01/out/source/starts.parquet'),
        ('benchmark', 'runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet')]
    counts = {}
    for label, path in sources:
        seen = set()
        for row in pq.ParquetFile(path).read().to_pylist():
            if row['seed_kind'] != 'human': continue
            start = row['start']; exact, composition, scenario = identities(start)
            f = families.setdefault(exact, dict(composition=composition, source_ids=set(), sources=set(), starts={},
                  baseline=dict(mcts20k=set(), puct=set()), files=set(), safe=compatible(start), trained=False))
            f['source_ids'].add(row['deck_id']); f['sources'].add(label); f['starts'][scenario] = start
            f['safe'] |= compatible(start); seen.add(exact)
        counts[label] = dict(exact_families=len(seen), compatible=sum(families[k]['safe'] for k in seen),
                             compositions=len({families[k]['composition'] for k in seen}))
    trained = json.loads(Path('runs/schema=gumbel_nn_combat_v1/date=2026-10-28/id=gumbel-nn-champ-fifty-20261028/out/inputs/starts.json').read_text())
    trained_compositions = {identities(row['start'])[1] for row in trained}
    for row in trained:
        exact, _, _ = identities(row['start'])
        if exact in families: families[exact]['trained'] = True
    variants = defaultdict(lambda: dict(rows=0, families=set(), scenarios=set(), files=set()))
    agents = Counter(); skipped = []; files = sorted(Path('runs/schema=combat_v4').rglob('fights*.parquet'))
    for i, path in enumerate(files):
        pf = pq.ParquetFile(path)
        if not {'agent', 'start'} <= set(pf.schema_arrow.names):
            skipped.append(str(path)); continue
        for batch in pf.iter_batches(columns=['agent', 'start'], batch_size=2048):
            for row in batch.to_pylist():
                agent = row['agent'] or ''; agents[agent] += 1
                kind = classify(agent)
                if not kind or 'a-gumbel' in str(path): continue
                start = row['start']
                if start['encounter'] != 39 or start['ascension'] != 20: continue
                exact, composition, scenario = identities(start)
                f = families.setdefault(exact, dict(composition=composition, source_ids=set(), sources=set(), starts={},
                      baseline=dict(mcts20k=set(), puct=set()), files=set(), safe=compatible(start), trained=False))
                f['starts'][scenario] = start; f['safe'] |= compatible(start)
                f['baseline'][kind].add((scenario, start['seed'])); f['files'].add(str(path))
                # Deliberately keep policy/run identities separate: PUCT files are NOT one common model.
                run = '/'.join(path.parts[:4]); variant = variants[(kind, run, agent)]
                variant['rows'] += 1; variant['families'].add(exact); variant['scenarios'].add(scenario); variant['files'].add(str(path))
        if i % 100 == 0: print(f'scanned {i+1}/{len(files)} parquet files', flush=True)
    manifest = []
    for exact, f in sorted(families.items()):
        a, b = f['baseline']['mcts20k'], f['baseline']['puct']
        manifest.append(dict(family=exact, composition=f['composition'], source_ids=sorted(f['source_ids']),
          sources=sorted(f['sources']), scenarios=len(f['starts']), safe=f['safe'], trained=f['trained'],
          composition_previously_trained=f['composition'] in trained_compositions,
          mcts20k_seed_starts=len(a), puct_seed_starts=len(b), matched_seed_starts=len(a & b),
          baseline_files=sorted(f['files']), example_start=next(iter(f['starts'].values()))))
    groups={}
    for label, subset in [('all',manifest),('compatible',[m for m in manifest if m['safe']]),
                         ('pool_compatible',[m for m in manifest if m['safe'] and 'pool' in m['sources']]),
                         ('unseen_composition_compatible',[m for m in manifest if m['safe'] and not m['composition_previously_trained']])]:
        groups[label]=dict(families=len(subset),compositions=len({m['composition'] for m in subset}),
          mcts20k=sum(m['mcts20k_seed_starts']>0 for m in subset), puct=sum(m['puct_seed_starts']>0 for m in subset),
          both=sum(m['mcts20k_seed_starts']>0 and m['puct_seed_starts']>0 for m in subset),
          matched=sum(m['matched_seed_starts']>0 for m in subset))
    report=dict(source_counts=counts,groups=groups,parquet_files=len(files),skipped_files=skipped,
                caution='Historical results use different simulators, models, search budgets and seed counts. Do not pool policy variants.',
                classifiers='MCTS20k; action-level PUCT2k with explore=0; excludes oracle, Gumbel, policy-only and turn-search variants',
                json_only_not_scanned=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'families.json').write_text(json.dumps(manifest,indent=2))
    policies=[dict(kind=k[0],run=k[1],agent=k[2],rows=v['rows'],families=len(v['families']),scenarios=len(v['scenarios']),files=sorted(v['files'])) for k,v in variants.items()]
    (OUT/'policies.json').write_text(json.dumps(sorted(policies,key=lambda v:-v['families']),indent=2))
    (OUT/'agents.json').write_text(json.dumps(agents,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
