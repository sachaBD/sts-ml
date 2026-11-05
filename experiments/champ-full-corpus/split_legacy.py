"""Split recorded legacy fights by deck familiarity without changing their bytes."""
import argparse
from collections import Counter
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
    parser.add_argument('legacy', type=Path, help='10-fight-per-deck legacy file')
    parser.add_argument('historical', type=Path, help='50-fight-per-deck historical validation file')
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    groups = {
        'legacy_core': [line for line in args.historical.read_bytes().splitlines(keepends=True)
                        if deck_of(json.loads(line)['fight_id']) in LEGACY_CORE],
        'legacy_extended': [line for line in args.legacy.read_bytes().splitlines(keepends=True)
                            if deck_of(json.loads(line)['fight_id']) not in LEGACY_CORE],
    }
    counts = Counter(deck_of(json.loads(line)['fight_id']) for line in groups['legacy_core'])
    if counts != {deck: 50 for deck in LEGACY_CORE}:
        parser.error('expected 50 historical fights for each core deck')
    args.out.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, lines in groups.items():
        path = args.out / f'{name}.jsonl'
        path.write_bytes(b''.join(lines))
        outputs.append(path)
        print(f'{name}: {len(lines)} fights')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.legacy, args.historical, *outputs]}
    (args.out / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()
