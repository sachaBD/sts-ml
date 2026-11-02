"""Freeze this experiment's composition-disjoint inputs; no outcome-based selection and no gameplay."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from apps.common.app import read_jsonl
from environments.combat.starts import Starts, fights
from apps.gumbel_nn_combat_expert_iteration.reporting import deck_of

NAMESPACE = 'champ-full-corpus-v1'


def composition(fight):
    return tuple(sorted(Counter(c['id'] for c in fight['start']['deck']).items()))


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def partition(pool, legacy):
    """950 new validation and 1,000 final compositions; legacy compositions never train. Keep all training variants,
    but just one representative per held-out composition; equivalent held-out variants are explicitly withheld."""
    groups = defaultdict(list)
    for f in pool:
        groups[composition(f)].append(f)
    legacy_comps = {composition(f) for f in legacy}
    if len(legacy) != 50 or len(legacy_comps) != 50:
        raise ValueError('expected 50 distinct legacy compositions')
    if not legacy_comps <= groups.keys():
        raise ValueError('legacy compositions missing from candidate pool')
    available = sorted(groups.keys() - legacy_comps, key=lambda c: digest([NAMESPACE, c]))
    if len(available) <= 1950:
        raise ValueError('not enough unseen compositions for validation, final and training')
    new, final = available[:950], available[950:1950]
    training = sorted((f for c in available[1950:] for f in groups[c]), key=lambda f: f['fight_id'])
    representative = lambda c: min(groups[c], key=lambda f: f['fight_id'])
    validation, reserved = [representative(c) for c in new], [representative(c) for c in final]
    withheld = [f for c in [*new, *final] for f in groups[c] if f != representative(c)]
    matches = {f['fight_id']: sorted(x['fight_id'] for x in groups[composition(f)]) for f in legacy}
    return training, validation, reserved, withheld, matches


def fixed_fights(decks, role, count):
    out = []
    for deck in decks:
        name = f'{NAMESPACE}:{deck["fight_id"]}:{role}'
        rng = random.Random(name)
        for f in fights(deck, name, count):
            Starts(potions=True).vary(f['start'], rng)
            out.append(f)
    return out


def legacy_fights(legacy, recorded):
    selected = []
    for d in legacy:
        available = [f for f in recorded if deck_of(f['fight_id']) == d['fight_id']]
        if len(available) < 10:
            raise ValueError(f'{d["fight_id"]}: fewer than ten recorded validation starts')
        selected.extend(sorted(available, key=lambda f: digest([NAMESPACE, f['fight_id']]))[:10])
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True, help='legacy directory with training.jsonl and validation.jsonl')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    pool, legacy = read_jsonl(a.pool), read_jsonl(a.legacy / 'training.jsonl')
    train, new, final, withheld, matches = partition(pool, legacy)
    legacy_eval = legacy_fights(legacy, read_jsonl(a.legacy / 'validation.jsonl'))
    files = {'training.jsonl': train, 'validation.jsonl': fixed_fights(new, 'validation', 2),
             'legacy.jsonl': legacy_eval, 'final.jsonl': fixed_fights(final, 'final', 10)}
    a.out.mkdir(parents=True, exist_ok=False)
    for name, rows in files.items():
        (a.out / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cohorts = {f['fight_id']: 'unseen_validation' for f in new} | {f['fight_id']: 'legacy50' for f in legacy}
    (a.out / 'cohorts.json').write_text(json.dumps(cohorts, indent=2) + '\n')
    selection = dict(training=[f['fight_id'] for f in train], validation=[f['fight_id'] for f in new],
                     final=[f['fight_id'] for f in final], legacy_matches=matches,
                     withheld_equivalents=[f['fight_id'] for f in withheld])
    (a.out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    summary = dict(candidate_decks=len(pool), candidate_compositions=len({composition(f) for f in pool}),
                   training_decks=len(train), training_compositions=len({composition(f) for f in train}),
                   new_validation_decks=len(new), legacy_decks=len(legacy), final_decks=len(final),
                   withheld_equivalents=len(withheld), fights={name: len(rows) for name, rows in files.items()})
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    inputs = [a.pool, a.legacy / 'training.jsonl', a.legacy / 'validation.jsonl', Path(__file__)]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [*inputs, *sorted(a.out.iterdir())]}
    (a.out / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
