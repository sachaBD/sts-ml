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
import hashlib
import json
from pathlib import Path
import subprocess

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
from .model import CONTRACT, NAMES, WIDTHS

WORKER = Path(__file__).resolve().parents[3] / 'build/pv/agents/combat/pv/pv_worker'
COUNTS = dict(zip(NAMES[1:], ('n_cards', 'n_monsters', 'n_potions', 'n_relics', 'n_actions')))


def collect(fights, searches, encounters, out, worker=WORKER):
    """Encode the selected fights into out/rows.parquet; returns the number of fights."""
    join = ''
    if searches:
        join = f"""left join (select fight_id, list({{'step': step, 'root_value': root_value, 'children': children}}
                   order by step) as search from read_parquet({[str(s) for s in searches]!r}) group by fight_id) s using (fight_id)"""
    db = duckdb.connect()
    db.execute("set memory_limit='2GB'; set threads=4")
    cursor = db.execute(f"""select f.fight_id, f.start, f.actions, f.won, f.final_hp,
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
            for fight_id, start, actions, won, hp, search in batch:
                proc.stdin.write(json.dumps(dict(fight_id=fight_id, start=start, actions=actions, won=won, final_hp=hp,
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


class Shard:
    """One Parquet shard in memory: flat float arrays plus per-row offsets; `batch` pads rows into torch tensors."""

    def __init__(self, path, valid):
        table = pq.read_table(path)
        if table.schema.metadata.get(b'pv_contract') != CONTRACT.encode():
            raise ValueError(f'{path}: input/value contract mismatch; regenerate it with agents.combat.pv.data')
        seeds = table['seed'].to_numpy()
        flags = {s: validation(s) for s in np.unique(seeds)}
        self.rows = np.flatnonzero(np.array([flags[s] for s in seeds], dtype=bool) == valid)
        self.value = table['value_target'].to_numpy().astype(np.float32)
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
        return out, torch.from_numpy(self.value[idx]), torch.from_numpy(policy), torch.from_numpy(self.has_policy[idx])


class Dataset:
    """Shards of one split. Default keeps them all in RAM; `stream` re-reads one shard at a time per epoch."""

    def __init__(self, paths, valid, stream=False):
        self.paths, self.valid, self.stream = list(paths), valid, stream
        self.loaded = None if stream else [Shard(p, valid) for p in self.paths]

    def batches(self, size, rng=None):
        """Yield (inputs, value, policy, has_policy); shards and rows are shuffled when `rng` is given."""
        order = np.arange(len(self.paths))
        if rng is not None: rng.shuffle(order)
        for k in order:
            shard = self.loaded[k] if self.loaded else Shard(self.paths[k], self.valid)
            rows = shard.rows.copy()
            if rng is not None: rng.shuffle(rows)
            for i in range(0, len(rows), size):
                yield shard.batch(rows[i:i + size])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fights', type=Path, nargs='+', required=True)
    p.add_argument('--search', type=Path, nargs='*', default=[])
    p.add_argument('--encounters', type=int, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', type=Path, default=WORKER)
    a = p.parse_args()
    print({'fights': collect(a.fights, a.search, a.encounters, a.out, a.worker)})
