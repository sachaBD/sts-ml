"""Step 3 gate: Gumbel root search at low sims vs stored PUCT-2k, same 400 dev starts, round-4 model, no training.

Reuses the corpus pilot's dev starts and its stored arms (no replay):
  PUCT2k  = out/dev-round4 (round-4 model, 2000 sims, rollout_mix .5, frozen worker 58c5c4ab)
  MCTS20k = out/dev-teacher
New arms play with build/a-gumbel pv_worker (greedy, gumbel_scale 0, m=16). Resumable via play_stage journals.

  .venv/bin/python experiments/25-10-26/a-gumbel/gate.py [--arms g256_r05,...] [--workers 10]
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from apps.combat_expert_iteration.corpus_io import atomic_json, journal_rows, play_stage

REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / 'runs/schema=combat_v4/date=2026-10-09/id=champ-corpus-pilot-v1/out'
OUT = REPO / 'runs/schema=combat_v4/date=2026-10-25/id=a-gumbel-gate-v1/out'
WORKER = REPO / 'build/a-gumbel/agents/combat/pv/pv_worker'
MODEL = PILOT / 'round4/model/model.onnx'
ARMS = {  # name -> (sims, extra flags)
    'g256_r05': (256, ['--rollout-mix', '0.5', '--root-halving']),
    'g256_r0': (256, ['--root-halving']),
    'g128_r05': (128, ['--rollout-mix', '0.5', '--root-halving']),
    'g128_r0': (128, ['--root-halving']),
    'puct256_r05': (256, ['--rollout-mix', '0.5']),  # control: is Gumbel better than just cutting PUCT sims?
}


def won(r):
    if r['status'] not in ('completed', 'capped'):
        raise ValueError('unresolved infrastructure error')
    return int(r['status'] == 'completed' and r['fight']['won'])


def compare(result, other, starts, rng_seed=20261025):
    """Equal-weight family mean paired difference, family-cluster bootstrap 95% CI (as corpus.evaluate)."""
    families = sorted({s['fight_id'].split(':')[1] for s in starts})
    diffs = [np.mean([won(result[s['fight_id']]) - won(other[s['fight_id']])
                      for s in starts if s['fight_id'].split(':')[1] == f]) for f in families]
    idx = np.random.default_rng(rng_seed).integers(len(families), size=(10000, len(families)))
    lo, hi = np.quantile(np.asarray(diffs)[idx].mean(1), [.025, .975])
    return dict(gap_pp=100 * float(np.mean(diffs)), ci95_pp=[100 * float(lo), 100 * float(hi)])


def cost(rows):
    decisions = [s['seconds'] for r in rows.values() for s in r.get('stats', [])]
    return dict(seconds_per_game=statistics.mean(r['seconds'] for r in rows.values()),
                decision_ms_median=1000 * statistics.median(decisions), decisions=len(decisions))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()
    starts = json.loads((PILOT / 'starts/development.json').read_text())
    starts = [dict(fight_id=s['fight_id'], start=s['start']) for s in starts]
    puct, mcts = journal_rows(PILOT / 'dev-round4/results.jsonl'), journal_rows(PILOT / 'dev-teacher/results.jsonl')
    report = dict(model=str(MODEL), worker=str(WORKER), n=len(starts), arms={},
                  puct2k=dict(wins=sum(map(won, puct.values())), **cost(puct)),
                  mcts20k=dict(wins=sum(map(won, mcts.values())), **cost(mcts)))
    for name in a.arms.split(','):
        sims, flags = ARMS[name]
        command = [str(WORKER), 'play', str(MODEL), str(sims)] + flags
        rows = play_stage(OUT / name, starts, command, a.workers)
        report['arms'][name] = dict(wins=sum(map(won, rows.values())), capped=sum(r['status'] == 'capped' for r in rows.values()),
                                    vs_puct2k=compare(rows, puct, starts), vs_mcts20k=compare(rows, mcts, starts), **cost(rows))
        atomic_json(OUT / 'report.json', report)
        print(json.dumps({name: report['arms'][name]}), flush=True)


if __name__ == '__main__':
    main()
