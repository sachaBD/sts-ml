"""Collect historical outcomes on the recorded Legacy Core fights; no gameplay."""
import argparse
import json
from pathlib import Path

from apps.common.app import read_jsonl
from apps.gumbel_nn_combat_expert_iteration.reporting import outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('core', type=Path)
    parser.add_argument('historical', type=Path, help='champ-50 historical baselines.json')
    parser.add_argument('gumbel', type=Path, help='champ50 specialist evaluation results.jsonl')
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    keys = {r['fight_id'] for r in read_jsonl(args.core)}
    historical = json.loads(args.historical.read_text())
    columns = {
        'PUCT u6 · 2k (historical)': historical['puct'],
        'MCTS20k (historical)': historical['mcts20k'],
        f'Gumbel specialist · {args.gumbel.parent.name}': outcomes(args.gumbel),
    }
    selected = {name: {k: values[k] for k in sorted(keys & values.keys())} for name, values in columns.items()}
    with args.out.open('x') as output:
        output.write(json.dumps(selected, indent=2) + '\n')


if __name__ == '__main__':
    main()
