"""Host-side replay buffer. Decisions wait per (lane, episode) until their game ends, then enter a ring buffer
with value target z = won (truncated or lost: 0)."""
from collections import defaultdict

import jax
import numpy as np


class Replay:
    def __init__(self, capacity: int, seed: int):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.data = None
        self.pending = defaultdict(list)   # (lane, episode) -> [(t, chunk arrays)]
        self.rng = np.random.default_rng(seed)
        self.games = self.wins = self.truncated = self.errors = 0

    def _store(self, rows: dict):
        n = len(rows["value"])
        if self.data is None:
            self.data = {k: np.zeros((self.capacity,) + v.shape[1:], v.dtype) for k, v in rows.items()}
        idx = (self.pos + np.arange(n)) % self.capacity
        for k, v in rows.items():
            self.data[k][idx] = v
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def add(self, rec):
        """rec: selfplay.Record with leading [T, B]."""
        rec = jax.device_get(rec)
        obs = {f"obs_{k}": np.asarray(v) for k, v in rec.obs._asdict().items()}
        cols = {**obs, "mask": np.asarray(rec.mask), "policy": np.asarray(rec.policy, np.float32)}
        T, B = rec.done.shape
        for b in range(B):
            for t in range(T):
                key = (b, int(rec.episode[t, b]))
                self.pending[key].append({k: v[t, b] for k, v in cols.items()})
                if rec.done[t, b]:
                    rows = self.pending.pop(key)
                    z = float(rec.won[t, b])
                    self.games += 1
                    self.wins += int(rec.won[t, b])
                    self.truncated += int(rec.truncated[t, b])
                    self.errors += int(rec.error[t, b] != 0)
                    stacked = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
                    stacked["value"] = np.full(len(rows), z, np.float32)
                    self._store(stacked)

    def sample(self, n: int) -> dict:
        idx = self.rng.integers(0, self.size, n)
        return {k: v[idx] for k, v in self.data.items()}


def batch_obs(batch: dict):
    """Rebuild a jaxsim.Obs (leading batch dim) from a sampled batch."""
    import jaxsim
    return jaxsim.Obs(**{k[4:]: batch[k] for k in batch if k.startswith("obs_")})
