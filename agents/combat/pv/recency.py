"""Fight-balanced recency replay; sampling only, no target or feature changes.

A fight's weight is 2**(-age / half_life), where age counts collection updates.
Each sampled fight contributes one uniformly sampled state. Shard/fight lengths
therefore do not bias fight selection. Explicit split required; validation never sampled.
"""
from collections import Counter
from pathlib import Path
import math

import numpy as np
from .data import Shard, merge


def probabilities(generations, current, half_life):
    generations = np.asarray(generations)
    if generations.ndim != 1 or not len(generations):
        raise ValueError('nonempty one-dimensional generations required')
    if not np.isfinite(generations).all() or not np.equal(generations, np.floor(generations)).all():
        raise ValueError('generations must be finite integers')
    if not isinstance(current, (int, np.integer)) or np.any(generations > current):
        raise ValueError('current must be integer and no generation may be in the future')
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError('half_life must be positive and finite')
    # Subtract maximum log weight to avoid underflow when every generation is old.
    logw = (generations.astype(float) - current) * math.log(2) / half_life
    w = np.exp(logw - logw.max())
    return w / w.sum()


class ReplayPool:
    def __init__(self, specs, split, current, half_life=None):
        if split is None:
            raise ValueError('explicit train/val split required')
        self.shards = []
        self.fights = []
        seen = set()
        for spec in specs:
            shard = Shard(Path(spec['path']), False, split)
            index = len(self.shards)
            self.shards.append(shard)
            for fid in np.unique(shard.fight_ids[shard.rows]):
                if fid in seen:
                    raise ValueError(f'duplicate training fight: {fid}')
                seen.add(fid)
                rows = shard.rows[shard.fight_ids[shard.rows] == fid]
                self.fights.append((str(fid), index, rows, spec['generation']))
        if not self.fights:
            raise ValueError('empty training pool')
        # Validate ages even for a uniform anchor pool.
        p = probabilities([f[3] for f in self.fights], current, half_life or 1.)
        if half_life is not None:
            if half_life <= 0:
                raise ValueError('half_life must be positive')
            self.p = p
        else:
            self.p = None  # Preserve original uniform integers() RNG behavior for anchors.
        self.realized = Counter()
        self.generation_draws = Counter()

    @property
    def ids(self):
        return {f[0] for f in self.fights}

    def expected_generation_share(self):
        out = Counter()
        p = self.p if self.p is not None else np.full(len(self.fights), 1 / len(self.fights))
        for f, prob in zip(self.fights, p):
            out[str(f[3])] += float(prob)
        return dict(out)

    def sample(self, rng, n):
        if n < 1:
            raise ValueError('positive sample count required')
        picks = rng.integers(len(self.fights), size=n) if self.p is None else rng.choice(len(self.fights), size=n, p=self.p)
        by = {}
        for k in picks:
            fid, index, rows, generation = self.fights[k]
            by.setdefault(index, []).append(rows[rng.integers(len(rows))])
            self.realized[fid] += 1
            self.generation_draws[str(generation)] += 1
        return merge([self.shards[i].batch(np.asarray(idx)) for i, idx in sorted(by.items())])
