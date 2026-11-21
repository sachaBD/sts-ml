"""Searches reference positions over a sims x worlds grid; one draw per cell, the same draw for every cell.

Worlds are drawn sequentially, so at one draw the smaller world sets are prefixes of the larger: cells along the
worlds axis share worlds (paired), and none share the reference's draws.

Usage: grid.py REFERENCE_DIR OUT_DIR POSITIONS SIMS,... WORLDS,...
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.combat_inspector import decisions, request
from apps.search_scaling.reference import CHECKPOINT, CHUNK, root

DRAW = 1


def main():
    ref, out = Path(sys.argv[1]), Path(sys.argv[2])
    count = int(sys.argv[3])
    sims = [int(x) for x in sys.argv[4].split(',')]
    worlds = [int(x) for x in sys.argv[5].split(',')]
    cells = [(s, w) for s in sims for w in worlds]
    positions = [json.loads(l) for l in open(ref / 'positions.jsonl')][:count]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / 'cells.jsonl', 'w') as f, open(out / 'progress.log', 'a') as progress:
        for first in range(0, len(positions), CHUNK):
            chunk = range(first, min(first + CHUNK, len(positions)))
            reqs = [request(positions[k]['start'], moves=positions[k]['moves'], sims=s, particles=w, world_draw=DRAW)
                    for k in chunk for s, w in cells]
            responses = iter(decisions(CHECKPOINT, reqs))
            for k in chunk:
                for s, w in cells:
                    f.write(json.dumps(dict(position=k, sims=s, worlds=w, **root(next(responses)))) + '\n')
            f.flush()
            progress.write(f'{time.time() - t0:7.0f}s {chunk.stop}/{len(positions)} positions\n'); progress.flush()
    (out / 'meta.json').write_text(json.dumps(dict(reference=str(ref), positions=count, sims=sims, worlds=worlds,
                                                   draw=DRAW, checkpoint=str(CHECKPOINT)), indent=1))


if __name__ == '__main__':
    main()
