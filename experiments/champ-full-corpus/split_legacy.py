"""Split recorded legacy fights by deck familiarity without changing their bytes."""
import argparse
import hashlib
import json
from pathlib import Path

from apps.gumbel_nn_combat_expert_iteration.reporting import deck_of

LEGACY_CORE = {
    'apparition-block', 'barricade-searing', 'clash', 'corruption-fnp', 'demon-combust',
    'exhume-immolate', 'fire-breathing', 'limit-break-rage', 'panache', 'perfected-strike',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    groups = {'legacy_core': [], 'legacy_extended': []}
    decks = set()
    for line in args.source.read_bytes().splitlines(keepends=True):
        deck = deck_of(json.loads(line)['fight_id'])
        decks.add(deck)
        groups['legacy_core' if deck in LEGACY_CORE else 'legacy_extended'].append(line)
    if missing := LEGACY_CORE - decks:
        parser.error(f'core decks missing from source: {sorted(missing)}')
    args.out.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, lines in groups.items():
        path = args.out / f'{name}.jsonl'
        path.write_bytes(b''.join(lines))
        outputs.append(path)
        print(f'{name}: {len(lines)} fights')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.source, *outputs]}
    (args.out / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()
