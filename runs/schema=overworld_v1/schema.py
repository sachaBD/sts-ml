"""Public macro-state traces of actual overworld play. Not full simulator snapshots.

Exact hidden simulator state belongs only in replay artifacts, never NN inputs.
Lookahead estimates are annotations; hypothetical branches are not actual transitions.
"""
import pyarrow as pa

NAME = 'overworld_v1'
def schema(fields, table):
    return pa.schema(fields, metadata={b'schema': NAME.encode(), b'table': table.encode()})
TABLES = {
    'runs': schema([
        ('run_key', pa.string()), ('seed', pa.uint64()), ('status', pa.string()),
        ('boss', pa.string()), ('floor', pa.int32()), ('final_hp', pa.int32()),
        ('record_json', pa.string()),  # actual macro trace, excluding NN estimates/exploration annotations
    ], 'runs'),
    'steps': schema([
        ('run_key', pa.string()), ('step_index', pa.int32()), ('kind', pa.string()),
        ('decision', pa.string()), ('fight_id', pa.string()),
        ('record_json', pa.string()),
    ], 'steps'),
    'agent': schema([
        ('run_key', pa.string()), ('step_index', pa.int32()), ('source', pa.string()),
        ('values_json', pa.string()),
    ], 'agent'),
}

def register_duckdb_views(db, root):
    for table in TABLES:
        glob = f'schema={NAME}/*/*/out/{table}-*.parquet'
        if any(root.glob(glob)):
            db.execute(f"create view {NAME}_{table} as select *, concat_ws('/', schema, date, id) as run_id "
                       f"from read_parquet('{root}/{glob}', hive_partitioning=true, hive_types_autocast=false, union_by_name=true)")
