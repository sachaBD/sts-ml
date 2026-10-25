"""Pre-declared panache tests (SINGLE_DECK.md): 300 fixed test seeds from champ-ten-test-u6-v1, greedy, paired.

Stored, not replayed: ten-deck PUCT2k update 6 (out/ten) and MCTS20k (out/mcts) on the same seeds.
New arms (build/a-gumbel worker), amended 25 Oct (SINGLE_DECK.md):
  gumbel_v1_u06_g256  Gumbel v1 update 6 (200 games/update), Gumbel 256   (equal games)
  gumbel_v2_u06_g256  Gumbel v2 update 6 (1400 games/update, 9 states/fight), Gumbel 256  (equal compute)
  puct_u06_p2k        single-deck PUCT control update 6, PUCT 2000 (like-for-like reference)
All rollout_mix .5. Paired difference over the 300 seeds, normal-approximation 95% CI.

  .venv/bin/python experiments/25-10-26/a-gumbel/test_panache.py [--workers 10]
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from apps.combat_expert_iteration.corpus_io import atomic_json, journal_rows, play_stage

REPO = Path(__file__).resolve().parents[3]
STORED = REPO / 'runs/schema=combat_v4/date=2026-10-09/id=champ-ten-test-u6-v1/out'
GUMBEL = REPO / 'runs/schema=combat_v4/date=2026-10-25/id=a-gumbel-panache-v1/out'
GUMBEL2 = REPO / 'runs/schema=combat_v4/date=2026-10-25/id=a-gumbel-panache-v2/out'
PUCT = REPO / 'runs/schema=combat_v4/date=2026-10-25/id=a-puct-panache-v1/out'
OUT = REPO / 'runs/schema=combat_v4/date=2026-10-25/id=a-gumbel-panache-test-v1/out'
WORKER = REPO / 'build/a-gumbel/agents/combat/pv/pv_worker'
G256 = ['256', '--rollout-mix', '0.5', '--root-halving']
P2K = ['2000', '--rollout-mix', '0.5']
ARMS = {
    'gumbel_v1_u06_g256': (GUMBEL / 'update006/model/model.onnx', G256),
    'gumbel_v2_u06_g256': (GUMBEL2 / 'update006/model/model.onnx', G256),
    'gumbel_v1_u13_g256': (GUMBEL / 'update013/model/model.onnx', G256),

}


def won(r):
    if r['status'] not in ('completed', 'capped'):
        raise ValueError(f'unresolved infrastructure error: {r["fight_id"]}')
    return int(r['status'] == 'completed' and r['fight']['won'])


def paired(a, b, ids):
    d = [won(a[i]) - won(b[i]) for i in ids]
    m = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    return dict(gap_pp=100 * m, ci95_pp=[100 * (m - 1.96 * se), 100 * (m + 1.96 * se)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--arms', default=','.join(ARMS))
    a = p.parse_args()
    starts = [dict(fight_id=s['fight_id'], start=s['start']) for s in json.loads((STORED / 'starts.json').read_text())['panache']]
    ids = [s['fight_id'] for s in starts]
    ref = {k: {i: r for i, r in journal_rows(STORED / k / 'results.jsonl').items() if i in set(ids)} for k in ('ten', 'mcts')}
    report = dict(n=len(ids), stored={k: sum(won(v[i]) for i in ids) for k, v in ref.items()}, arms={})
    for name in a.arms.split(','):
        model, flags = ARMS[name]
        rows = play_stage(OUT / name, starts, [str(WORKER), 'play', str(model)] + flags, a.workers)
        report['arms'][name] = dict(wins=sum(won(rows[i]) for i in ids), capped=sum(rows[i]['status'] == 'capped' for i in ids),
                                    vs_ten_puct2k=paired(rows, ref['ten'], ids), vs_mcts20k=paired(rows, ref['mcts'], ids),
                                    seconds_per_game=statistics.mean(rows[i]['seconds'] for i in ids))
        atomic_json(OUT / 'report.json', report)
        print(json.dumps({name: report['arms'][name]}), flush=True)
    arms = {k: journal_rows(OUT / k / 'results.jsonl') for k in report['arms']}
    for k in ('gumbel_v1_u06_g256', 'gumbel_v2_u06_g256'):
        if {k, 'puct_u06_p2k'} <= arms.keys():
            report[f'{k}_vs_puct_u06_p2k'] = paired(arms[k], arms['puct_u06_p2k'], ids)
    atomic_json(OUT / 'report.json', report)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
