"""Small corpus stage primitives: immutable inputs, resumable journals, atomic outputs."""
import hashlib
import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from apps.common.app import sha256
from apps.common.worker import run_parallel
from apps.human_champ.bench import play_one
from apps.megacrit_dump.schema import load


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'immutable input mismatch: {path}')
    else:
        atomic_json(path, value)


def journal_rows(path, repair=False):
    """Only a torn final line can be repaired on explicit stage resume; interior corruption fails."""
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    rows, offset = {}, 0
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1 or not repair:
                raise ValueError(f'corrupt journal: {path}:{i+1}')
            with path.open('r+b') as stream:
                stream.truncate(offset)
            break
        if row['fight_id'] in rows:
            raise ValueError(f'duplicate result: {row["fight_id"]}')
        rows[row['fight_id']] = row
        offset += len(line)
    return rows


def play_stage(directory, starts, command, workers, progress=None):
    """Cache identity includes full starts, worker/model bytes and complete search command.

    Completed or agent-capped games are never replayed. Infrastructure errors halt the stage;
    retries require explicit review, rather than silently resampling or dropping pairs.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {command[0]: sha256(Path(command[0]))}
    if command[1] == 'play':
        model = Path(command[2])
        files[str(model)] = sha256(model)
        external = model.with_name(model.name + '.data')
        if external.exists():
            files[str(external)] = sha256(external)
    spec = dict(starts=starts, command=command, files=files, timeout=900)
    freeze_json(directory / 'stage.json', spec)
    rows = journal_rows(directory / 'results.jsonl', repair=True)
    expected = {r['fight_id'] for r in starts}
    if len(expected) != len(starts) or not set(rows) <= expected:
        raise ValueError('duplicate starts or unexpected cached results')
    if any(r['status'] not in ('completed', 'capped') for r in rows.values()):
        raise RuntimeError(f'cached infrastructure error requires review: {directory}')
    with (directory / 'results.jsonl').open('a') as journal:
        def done(_, result):
            journal.write(json.dumps(result) + '\n')
            journal.flush()
            rows[result['fight_id']] = result
            if progress:
                progress(dict(done=len(rows), total=len(starts), wins=sum(
                    r['status'] == 'completed' and r['fight']['won'] for r in rows.values()),
                    statuses={s: sum(r['status'] == s for r in rows.values())
                              for s in ('completed', 'capped', 'error')}))
            if result['status'] not in ('completed', 'capped'):
                raise RuntimeError(f'worker error: {result}')
        run_parallel(play_one, [(command, r, 900) for r in starts if r['fight_id'] not in rows], workers, done)
    completed = [r for r in rows.values() if r['status'] == 'completed']
    schema = load('combat_v4')
    for name, values, arrow in (
        ('fights', [r['fight'] for r in completed], schema.FIGHTS),
        ('search', [s for r in completed for s in r.get('search', [])], schema.SEARCH),
    ):
        path = directory / f'{name}.parquet'
        tmp = path.with_suffix('.tmp')
        pq.write_table(pa.Table.from_pylist(values, schema=arrow), tmp)
        os.replace(tmp, path)
    atomic_json(directory / 'done.json', dict(count=len(rows), identity=identity(spec), time=time.time()))
    return rows
