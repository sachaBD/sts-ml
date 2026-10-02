"""Atomic replay facts and separate agent/derived annotations for joint self-play."""
import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

@cache
def contract(name):
    path = Path(__file__).resolve().parents[2] / 'runs' / f'schema={name}' / 'schema.py'
    spec = importlib.util.spec_from_file_location(f'runs.{name}_schema', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def atomic(out, table, key, rows, schema):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / f'.{table}-{key}.tmp'
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression='zstd')
    os.replace(tmp, out / f'{table}-{key}.parquet')

def write_combat(out, records, key):
    tables = contract('combat_v4').TABLES
    rows = {t: [] for t in tables}
    for rec in records:
        fid = rec['fight_id']
        if rec['initial_state'] is not None:
            rows['fights'].append({'fight_id': fid, 'schema_version': 1, 'initial_state': rec['initial_state']})
        for row in rec['rows']:
            if row['row_kind'] == 'decision':
                i = row['decision_index']
                rows['steps'].append({'fight_id': fid, 'step_index': i, 'action_bits': row['executed_action_bits']})
                rows['search'].append({'fight_id': fid, 'step_index': i, 'root_value': row['root_value'],
                    'simulations_used': row['simulations_used'], 'explored': row['was_random'],
                    'actions_json': json.dumps(row['actions'])})
            rows['training'].append({**row, 'fight_id': fid})
        rows['results'].append({**rec['result'], 'fight_id': fid})
    for name, table in tables.items():
        atomic(out, table.file_prefix, key, rows[name], table.schema)

def write_overworld(out, msg, key):
    tables = contract('overworld_v1').TABLES
    facts = dict(msg); facts['steps'] = []
    steps, annotations = [], []
    for i, original in enumerate(msg['steps']):
        step = dict(original)
        source, values = step.pop('source', None), step.pop('values', None)
        facts['steps'].append(step)
        steps.append({'run_key': msg['run_key'], 'step_index': i, 'kind': step['kind'],
            'decision': step.get('decision'), 'fight_id': step.get('fight_id'), 'record_json': json.dumps(step)})
        if source is not None:
            annotations.append({'run_key': msg['run_key'], 'step_index': i, 'source': source,
                                'values_json': json.dumps(values)})
    run = {k: msg[k] for k in ('run_key', 'seed', 'status', 'boss', 'floor', 'final_hp')}
    run['record_json'] = json.dumps(facts)
    for table, rows in [('runs', [run]), ('steps', steps), ('agent', annotations)]:
        atomic(out, table, key, rows, tables[table])
