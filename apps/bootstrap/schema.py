"""Stable PyArrow layout for combat_v1 parquet parts (see runs/README.md)."""

import pyarrow as pa

F32 = pa.float32()
COMBAT_V1 = pa.schema(
    [
        ("episode_id", pa.int64()),
        ("decision_index", pa.int32()),
        ("turn", pa.int32()),
        ("entry_id", pa.string()),
        ("deck_signature", pa.string()),
        ("combat_seed", pa.uint64()),
        ("starting_hp", pa.int16()),
        ("starting_max_hp", pa.int16()),
        ("encoding_version", pa.int32()),
        ("global_numeric", pa.list_(F32, 50)),
        ("cards", pa.list_(pa.struct([("card_id", pa.int16()), ("zone", pa.int8()), ("card_type", pa.int8()),
                                      ("target_type", pa.int8()), ("numeric", pa.list_(F32, 14))]))),
        ("monsters", pa.list_(pa.struct([("monster_id", pa.int16()), ("move_id", pa.int16()),
                                         ("numeric", pa.list_(F32, 9))]))),
        ("card_monster_interactions", pa.list_(pa.struct([("card_index", pa.int16()), ("monster_index", pa.int8()),
                                                          ("numeric", pa.list_(F32, 6))]))),
        ("input_state", pa.int16()),
        ("card_selection_task", pa.int16()),
        ("actions", pa.list_(pa.struct([("action", pa.int32()), ("description", pa.string()),
                                        ("visits", pa.int64()), ("mean_value", F32)]))),
        ("chosen_action", pa.int32()),
        ("was_random", pa.bool_()),
        ("root_value", F32),
        ("won", pa.bool_()),
        ("final_hp", pa.int16()),
        ("potions", pa.int8()),
        ("terminal_value", F32),
        ("row_kind", pa.string()),
        ("parent_action", pa.int32()),
        ("simulations_used", pa.int64()),
    ]
)
