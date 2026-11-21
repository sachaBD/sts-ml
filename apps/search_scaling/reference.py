"""Builds the reference regret set: one decision per fight, searched at 2048 sims x 64 worlds on two draws.

Writes root summaries to reference.jsonl and every full search graph to graphs.jsonl.gz.

Usage: reference.py OUT_DIR [FIGHTS]
"""
import gzip, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.combat_inspector import decisions, request

CHECKPOINT = Path('/home/sborowsk/project/sts/worktrees/graph-cycle-reporting/runs/schema=gumbel_nn_combat_v1/'
                  'date=2026-11-11/id=champ-full-corpus-v2-inventory48-wide128-v512-256s-graph-cyclefix/out/training/iter030')
PLAY = dict(sims=256, particles=8, gumbel_scale=0)  # the agent that walks the fights, with play's world draw
REFERENCE = dict(sims=2048, particles=64, considered_actions=128)
REFERENCE_DRAWS = (1_000_001, 1_000_002)
CHUNK = 100  # positions per inspector call, so graphs stream to disk


def root(response):
    graph = response['graph']
    node = next(n for n in graph['nodes'] if n['observation'] == graph['root'])
    return dict(value=response['decision']['value'], action=response['decision']['action'],
                v_net=node['evaluation_value'], network_requests=response['network_requests'],
                edges=[{k: e[k] for k in ('bits', 'visits', 'q', 'logit', 'prior', 'improved_policy')}
                       for e in node['edges']])


def walk(starts, log):
    """Plays every fight with PLAY; returns, per fight, the move lists before each decision with 2+ actions."""
    moves = [[] for _ in starts]
    choices = [[] for _ in starts]
    live = list(range(len(starts)))
    while live:
        out = decisions(CHECKPOINT, [request(starts[i], moves=moves[i], **PLAY) for i in live])
        still = []
        for i, r in zip(live, out):
            if r['over']:
                continue
            if len(root(r)['edges']) >= 2:
                choices[i].append(list(moves[i]))
            moves[i].append(r['decision']['action'])
            still.append(i)
        live = still
        log(f'walk: {len(live)} fights live')
    return choices


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fights = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    t0 = time.time()
    with open(out / 'progress.log', 'a') as progress:
        def log(message):
            progress.write(f'{time.time() - t0:7.0f}s {message}\n'); progress.flush()
        records = [json.loads(l) for l in open(CHECKPOINT / 'starts.jsonl')]
        records = random.Random(0).sample(records, fights)
        starts = [r['start'] for r in records]
        choices = walk(starts, log)
        positions = []
        for i, options in enumerate(choices):
            if options:
                positions.append(dict(fight_id=records[i]['fight_id'], start=starts[i],
                                      moves=random.Random(i).choice(options), decisions=len(options)))
        (out / 'positions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in positions))
        log(f'positions: {len(positions)} from {fights} fights')
        with open(out / 'reference.jsonl', 'w') as summaries, gzip.open(out / 'graphs.jsonl.gz', 'wt') as graphs:
            for first in range(0, len(positions), CHUNK):
                chunk = range(first, min(first + CHUNK, len(positions)))
                reqs = [request(positions[k]['start'], moves=positions[k]['moves'], world_draw=d, **REFERENCE)
                        for k in chunk for d in REFERENCE_DRAWS]
                responses = decisions(CHECKPOINT, reqs)
                for j, k in enumerate(chunk):
                    pair = responses[2 * j: 2 * j + 2]
                    summaries.write(json.dumps(dict(position=k, draws=[root(r) for r in pair])) + '\n')
                    for d, r in zip(REFERENCE_DRAWS, pair):
                        graphs.write(json.dumps(dict(position=k, world_draw=d, response=r)) + '\n')
                summaries.flush()
                log(f'reference: {chunk.stop}/{len(positions)} positions')
        (out / 'meta.json').write_text(json.dumps(dict(checkpoint=str(CHECKPOINT), play=PLAY, reference=REFERENCE,
                                                       draws=REFERENCE_DRAWS, fights=fights), indent=1))
        log('reference: done')


if __name__ == '__main__':
    main()
