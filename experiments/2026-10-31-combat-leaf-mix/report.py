"""Read completed sweep outputs; never launch fights. Writes report.json and report.md beside them."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import outcomes, save
from apps.common.app import read_jsonl
from apps.gumbel_nn_combat_expert_iteration.reporting import by_deck, deck_of, paired, score


def summarize(results, references):
    if set(results) != set(references['nn50']):
        raise ValueError('results must match all frozen reference IDs')
    decks = by_deck(results)
    for deck, row in decks.items():
        keys = [k for k in results if deck == 'all' or deck_of(k) == deck]
        row['vs_nn50'] = paired(results, references['nn50'], keys)
    return dict(aggregate=score(results, references), decks=decks)


def render(summaries):
    lines = ['# Checkpoint-132 leaf-mix sweep', '',
             '500 paired frozen starts, 50/deck. Trained-on-deck monitoring set, not a held-out test.',
             '95% intervals: Wilson for win rates; normal paired-difference intervals over fights.',
             'Aggregate JSON also includes deck-level variability (10 decks); no multiplicity correction.',
             'Historical MCTS/PUCT simulator versions differ. Their gaps are descriptive, not causal.',
             'Only two new variants; one timing observation each, no timing uncertainty estimate.', '',
             '| Variant | Wins / 500 | Win rate [95% CI] | Δ vs NN50 [95% CI], pp | Process wall s |',
             '|---|---:|---|---|---:|']
    for name, summary in summaries.items():
        s = summary['aggregate']
        lo, hi = s['wilson_95']
        gap = s['comparisons']['nn50']
        gl, gh = gap['gap_95']
        timing = summary.get('process_wall_seconds')
        wall = f'{timing:.2f}' if timing is not None else 'historical / not comparable'
        lines.append(f'| {name} | {s["wins"]} | {100*s["win_rate"]:.1f}% [{100*lo:.1f}, {100*hi:.1f}] | '
                     f'{100*gap["gap"]:+.1f} [{100*gl:.1f}, {100*gh:.1f}] | {wall} |')
    lines += ['', '## Per deck (wins / 50)', '', '| Deck | ' + ' | '.join(summaries) + ' |',
              '|---|' + '---:|' * len(summaries)]
    for deck in summaries['nn50']['decks']:
        if deck != 'all':
            lines.append('| ' + deck + ' | ' + ' | '.join(str(s['decks'][deck]['w']) for s in summaries.values()) + ' |')
    lines += ['', 'Per-deck paired differences and intervals, and historical baseline comparisons, are in report.json.',
              'Timing includes native-process setup and gameplay; excludes preflight hashes and reporting.',
              'TensorRT cache state and host load can affect timing. Do not infer precise speedups from the cached baseline.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    references = json.loads((args.out / 'references.json').read_text())
    summaries = {name: summarize(rows, references) for name, rows in references.items()}
    for name in ('nn75', 'nn100'):
        directory = args.out / name
        status = json.loads((directory / 'status.json').read_text())
        if status['status'] != 'complete':
            raise ValueError(f'{name} is not complete; refusing partial report')
        summaries[name] = summarize(outcomes(read_jsonl(directory / 'results.jsonl')), references)
        summaries[name]['process_wall_seconds'] = status['process_wall_seconds']
    save(args.out / 'report.json', summaries)
    (args.out / 'report.md').write_text(render(summaries))
    print(args.out / 'report.md')


if __name__ == '__main__':
    main()
