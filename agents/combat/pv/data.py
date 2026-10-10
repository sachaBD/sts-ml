"""combat_v4 → PV training shards (Parquet), and shard → padded torch batches.

`python -m agents.combat.pv.data --fights F.parquet.. [--search S.parquet..] --encounters 39 --out DIR`
selects fights with DuckDB (2 GB, 4 threads), pipes them as JSON lines (transport only) to one `pv_worker encode`, which
replays every fight with the canonical native encoder and writes DIR/rows.parquet (zstd). A replay mismatch aborts;
nothing is published for a failed encode.

Shard schema, one row per decision: fight_id, seed (start.seed), encounter, step, turn, won, final_hp, value_target
(CombatObjective terminal value), root_value (teacher root value, null without search), moves list<uint32> (legal
Action::bits, encoder order), policy_target list<float32> (normalized visits, aligned with moves), has_policy,
the six model inputs flattened as list<float32> (context, cards, monsters, potions, relics, actions; row widths in
model.WIDTHS) and n_cards, n_monsters, n_potions, n_relics, n_actions row counts. File metadata pv_contract = model.CONTRACT.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from .model import CONTRACT, NAMES, WIDTHS

WORKER = Path(__file__).resolve().parents[3] / 'build/pv/agents/combat/pv/pv_worker'
COUNTS = dict(zip(NAMES[1:], ('n_cards', 'n_monsters', 'n_potions', 'n_relics', 'n_actions')))


def collect(fights, searches, encounters, out, worker=WORKER):
    """Encode the selected fights into out/rows.parquet; returns the number of fights."""
    join = ''
    if searches:
        # The search agent is carried through so evaluation-only root_halving records fail closed in collect and encode.
        join = f"""left join (select fight_id, list({{'step': step, 'root_value': root_value, 'children': children, 'agent': agent}}
                   order by step) as search from read_parquet({[str(s) for s in searches]!r}) group by fight_id) s using (fight_id)"""
    db = duckdb.connect()
    db.execute("set memory_limit='2GB'; set threads=1")
    cursor = db.execute(f"""select f.fight_id, f.start, f.actions, f.won, f.final_hp, f.agent,
        {'s.search' if searches else 'null'} as search from read_parquet({[str(f) for f in fights]!r}) f {join}
        where f.start.encounter in ({','.join(str(int(e)) for e in encounters)}) 
        order by f.fight_id""")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f'rows.parquet'
    partial = final.with_suffix('.tmp')
    count = 0
    with subprocess.Popen([str(worker), 'encode', str(partial)], stdin=subprocess.PIPE, text=True) as proc:
        while batch := cursor.fetchmany(128):
            for fight_id, start, actions, won, hp, agent, search in batch:
                if 'root_halving' in (agent or '') or any('root_halving' in (r.get('agent') or '') for r in search or []):
                    proc.stdin.close(); proc.kill()
                    raise RuntimeError(f'{fight_id}: root_halving records are evaluation-only, not training rows')
                proc.stdin.write(json.dumps(dict(fight_id=fight_id, start=start, actions=actions, won=won, final_hp=hp, agent=agent,
                                                 search=search or [])) + '\n')
                count += 1
        proc.stdin.close()
    if proc.returncode:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'pv_worker encode failed')
    partial.replace(final)
    return count


def validation(seed):
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], 'little') % 10 == 0


def load_split(path):
    """Explicit fight membership, assigned by source deck before seeds/descendants are made."""
    if path is None:
        return None
    mapping = json.loads(Path(path).read_text())
    if not isinstance(mapping, dict) or any(v not in ('train', 'val') for v in mapping.values()):
        raise ValueError('split manifest must map every fight_id to train or val')
    return mapping


class Shard:
    """One Parquet shard in memory: flat float arrays plus per-row offsets; `batch` pads rows into torch tensors."""

    def __init__(self, path, valid, split=None, train_fights=None, *, row_indices=None):
        table = pq.read_table(path)
        if row_indices is not None:
            table = table.take(row_indices)
        if table.schema.metadata.get(b'pv_contract') != CONTRACT.encode():
            raise ValueError(f'{path}: input/value contract mismatch; regenerate it with agents.combat.pv.data')
        if split is None:
            seeds = table['seed'].to_numpy()
            flags = {s: validation(s) for s in np.unique(seeds)}
            membership = [flags[s] for s in seeds]
        else:
            ids = table['fight_id'].to_pylist()
            missing = set(ids) - split.keys()
            if missing:
                raise ValueError(f'{path}: fights missing from split manifest: {sorted(missing)[:5]}')
            membership = [split[fid] == 'val' for fid in ids]
        self.fight_ids = np.asarray(table['fight_id'].to_pylist())
        selected = np.array(membership, dtype=bool) == valid
        if not valid and train_fights is not None:
            selected &= np.isin(self.fight_ids, list(train_fights))
        self.rows = np.flatnonzero(selected)
        self.value = 100 * table['won'].to_numpy().astype(np.float32)
        root = table['root_value'].to_numpy(zero_copy_only=False)
        self.root_value = np.array([np.nan if v is None else v for v in root], np.float32) if root.dtype == object \
            else root.astype(np.float32)
        self.has_policy = table['has_policy'].to_numpy(zero_copy_only=False)
        self.counts = {n: table[c].to_numpy() for n, c in COUNTS.items()}
        self.flat = {}
        for name in NAMES:
            column = table[name].combine_chunks()
            self.flat[name] = column.values.to_numpy(), column.offsets.to_numpy()
        policy = table['policy_target'].combine_chunks()
        self.policy = policy.values.to_numpy(), policy.offsets.to_numpy()

    def batch(self, idx):
        def rows(flat, r, width):
            values, offsets = flat
            return values[offsets[r]:offsets[r + 1]].reshape(-1, width)

        out = {'context': torch.from_numpy(np.stack([rows(self.flat['context'], r, 65)[0] for r in idx]))}
        for name, width in list(zip(NAMES, WIDTHS))[1:]:
            # >=2 rows keeps torch.export's dynamic dimensions unspecialized during export.
            count = max(2, self.counts[name][idx].max())
            x = np.zeros((len(idx), count, width), np.float32)
            for i, r in enumerate(idx):
                block = rows(self.flat[name], r, width)
                x[i, :len(block)] = block
            out[name] = torch.from_numpy(x)
        policy = np.zeros((len(idx), out['actions'].shape[1]), np.float32)
        for i, r in enumerate(idx):
            p = self.policy[0][self.policy[1][r]:self.policy[1][r + 1]]
            policy[i, :len(p)] = p
        return (out, torch.from_numpy(self.value[idx]), torch.from_numpy(policy), torch.from_numpy(self.has_policy[idx]),
                torch.from_numpy(self.root_value[idx]))


def merge(parts):
    """Concatenate batches from different shards, zero-padding token/action counts to the largest."""
    if len(parts) == 1:
        return parts[0]

    def cat(tensors):
        if tensors[0].dim() < 2:
            return torch.cat(tensors)
        n = max(t.shape[1] for t in tensors)
        return torch.cat([F.pad(t, (0, 0) * (t.dim() - 2) + (0, n - t.shape[1])) for t in tensors])

    return ({k: cat([p[0][k] for p in parts]) for k in parts[0][0]}, *(cat([p[i] for p in parts]) for i in range(1, 5)))


class Dataset:
    """Shards of one split. Default keeps them all in RAM; `stream` re-reads one shard at a time per epoch.

    `mix` (non-stream only) pools the sampled (shard, row) pairs of ALL shards and builds each minibatch across shards.
    `realized` counts the states served per fight in the latest pass over the data.
    """

    def __init__(self, paths, valid, stream=False, split=None, train_fights=None, states_per_fight=None, mix=False):
        self.paths, self.valid, self.stream = list(paths), valid, stream
        self.split, self.train_fights = split, train_fights
        self.states_per_fight = None if valid else states_per_fight
        self.mix, self.realized = mix, collections.Counter()
        if mix and stream:
            raise ValueError('mix-shards requires non-stream loading')
        self.loaded = None if stream else [Shard(p, valid, split, train_fights) for p in self.paths]
        if self.loaded:  # a fight owned by two included shards would be silently double counted
            seen = {}
            for path, shard in zip(self.paths, self.loaded):
                for fid in np.unique(shard.fight_ids[shard.rows]):
                    if fid in seen:
                        raise ValueError(f'fight {fid} appears in more than one shard: {seen[fid]} and {path}')
                    seen[fid] = path

    def _sample(self, shard, rng):
        rows = shard.rows.copy()
        if self.states_per_fight and len(rows):
            sampler = rng if rng is not None else np.random.default_rng(0)
            sampled = [sampler.choice(rows[shard.fight_ids[rows] == fid], self.states_per_fight, replace=True)
                       for fid in np.unique(shard.fight_ids[rows])]
            rows = np.concatenate(sampled)
        ids, counts = np.unique(shard.fight_ids[rows], return_counts=True)
        self.realized.update(dict(zip(ids.tolist(), counts.tolist())))
        return rows

    def batches(self, size, rng=None):
        """Yield (inputs, value, policy, has_policy, root_value [NaN when absent]); shards and rows are shuffled when `rng` is given."""
        self.realized = collections.Counter()
        if self.mix:
            yield from self._mixed(size, rng)
            return
        order = np.arange(len(self.paths))
        if rng is not None: rng.shuffle(order)
        for k in order:
            shard = self.loaded[k] if self.loaded else Shard(self.paths[k], self.valid, self.split, self.train_fights)
            rows = self._sample(shard, rng)
            if rng is not None: rng.shuffle(rows)
            for i in range(0, len(rows), size):
                yield shard.batch(rows[i:i + size])

    def _mixed(self, size, rng):
        pairs = [np.stack([np.full(len(rows), k), rows], 1)
                 for k, shard in enumerate(self.loaded) for rows in [self._sample(shard, rng)]]
        pairs = np.concatenate(pairs)
        if rng is not None: rng.shuffle(pairs)
        for i in range(0, len(pairs), size):
            chunk = pairs[i:i + size]
            yield merge([self.loaded[k].batch(chunk[chunk[:, 0] == k, 1]) for k in np.unique(chunk[:, 0])])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fights', type=Path, nargs='+', required=True)
    p.add_argument('--search', type=Path, nargs='*', default=[])
    p.add_argument('--encounters', type=int, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', type=Path, default=WORKER)
    a = p.parse_args()
    print({'fights': collect(a.fights, a.search, a.encounters, a.out, a.worker)})
