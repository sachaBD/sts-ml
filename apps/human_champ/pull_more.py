"""Bounded additional 2020 pull, reusing megacrit_dump; exclude previously downloaded files."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from apps.megacrit_dump.pull import choose_files, run


def select(files, excluded, n):
    rows = [r for r in files.to_pylist() if str(r['folder_path']).startswith('Monthly_2020')
            and r['file_id'] not in excluded]
    return choose_files(pa.Table.from_pylist(rows, files.schema), n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files', type=Path, required=True)
    p.add_argument('--exclude-runs', type=Path, nargs='+', required=True)
    p.add_argument('--n-files', type=int, default=800)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(a.files)
    excluded = set()
    for directory in a.exclude_runs:
        for file in directory.glob('out/done/part-*.parquet'):
            excluded.update(pq.read_table(file, columns=['file_id'])['file_id'].to_pylist())
    manifest = a.out / 'files-selected.parquet'
    if manifest.exists():
        files = pq.read_table(manifest).to_pylist()
    else:
        files = select(table, excluded, a.n_files)
        pq.write_table(pa.Table.from_pylist(files, table.schema), manifest, compression='zstd')
    print(f'{len(files)} fresh 2020 files; {len(excluded)} previous files excluded', flush=True)
    summary = run(files, a.out, 'IRONCLAD', 20, max(1, min(a.workers, 4)),
                  log=lambda message: print(message, flush=True))
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(3 if summary['throttled'] else 1 if summary['files_failed_this_invocation'] else 0)


if __name__ == '__main__':
    main()
