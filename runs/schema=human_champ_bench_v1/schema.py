"""One-off human-derived Ironclad A20 Champ benchmark, two starts per retained deck.

out/starts.parquet: joins agent combat_v4 fights on fight_id. Human outcome is repeated,
not two observations. Decks are exact under megacrit reconstruction; full historical
battle state is NOT exact. See description.md for approximations and exclusions.
"""
import importlib.util
from pathlib import Path
import pyarrow as pa

NAME = 'human_champ_bench_v1'
spec = importlib.util.spec_from_file_location('combat_v4_schema', Path(__file__).parents[1] / 'schema=combat_v4/schema.py')
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
STARTS = pa.schema([
    ('fight_id', pa.string()), ('deck_id', pa.string()), ('source_run_id', pa.string()),
    ('build_version', pa.string()), ('human_won', pa.bool_()), ('seed_kind', pa.string()),
    ('start', v4.START), ('demon_form', pa.bool_()), ('scaling_count', pa.int32()),
    ('hp_band', pa.string()), ('changed_cards', pa.list_(pa.string())),
    ('reset_counters', pa.list_(pa.string())),
], metadata={b'schema': NAME.encode(), b'table': b'starts'})
TABLES = {'starts': STARTS}
