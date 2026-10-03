"""combat_v4 → replayed model inputs/targets. JSONL is a disposable cache, not a new replay schema."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import duckdb
import torch
from .model import NAMES, WIDTHS


def collate(rows):
    result = {}
    for name, width in zip(NAMES, WIDTHS):
        if name == 'context':
            result[name] = torch.tensor([r['inputs'][name] for r in rows], dtype=torch.float32)
        else:
            # >=2 rows keeps torch.export's dynamic dimensions unspecialized during export.
            count = max(2, max(len(r['inputs'][name]) for r in rows))
            result[name] = torch.zeros(len(rows), count, width)
            for i, r in enumerate(rows):
                x = torch.tensor(r['inputs'][name], dtype=torch.float32)
                result[name][i, :len(x)] = x
    return result


def collect(dirs, worker, out, limit=None):
    fights = [str(Path(d) / 'fights-*.parquet') for d in dirs]
    searches = [str(Path(d) / 'search-*.parquet') for d in dirs]
    query = f"""with s as (
        select fight_id, list({{'step': step, 'children': children}} order by step) as search
        from read_parquet({searches!r}) group by fight_id)
        select f.fight_id, f.start, f.actions, f.won, f.final_hp, s.search
        from read_parquet({fights!r}) f left join s using (fight_id)
        where f.start.encounter = 39 order by f.fight_id"""  # sts_lightspeed CHAMP
    if limit: query += f' limit {int(limit)}'
    records = duckdb.sql(query).fetchall()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Commit the cache only after the native replay/encoder succeeds.
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        source = Path(tmp) / 'fights.jsonl'
        with source.open('w') as f:
            for fight_id, start, actions, won, hp, search in records:
                f.write(json.dumps(dict(fight_id=fight_id, start=start, actions=actions, won=won, final_hp=hp,
                                       search=search or [])) + '\n')
        encoded = Path(tmp) / 'rows.jsonl'
        with source.open() as stdin, encoded.open('w') as stdout:
            subprocess.run([str(worker), 'encode'], stdin=stdin, stdout=stdout, check=True)
        encoded.replace(out)
    return len(records)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combat', type=Path, nargs='+', required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    print({'fights': collect(a.combat, a.worker, a.out, a.limit), 'cache': str(a.out)})
