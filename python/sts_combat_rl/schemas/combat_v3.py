"""combat_v3: the table apps/bootstrap writes. Column meanings: runs/README.md ("combat_v3 columns").

NAME is the single source of the schema name: run.sh launches under it, and every parquet part and
summary.json carries it. Renaming, removing or changing the meaning of a column means a new NAME.
"""

import pyarrow as pa

NAME = "combat_v3"
F32 = pa.float32()
COMBAT_V3 = pa.schema(
    [
        # which run / fight
        ("run_seed", pa.uint64()),
        ("episode_id", pa.int64()),  # one fight: run_seed * 100 + fight_index
        ("fight_index", pa.int16()),
        ("source_episode_id", pa.int64()),  # apps/fight_resample: the stored fight this one resamples; NULL = a played run's own fight
        ("act", pa.int8()),
        ("floor", pa.int16()),
        ("encounter", pa.string()),
        ("category", pa.string()),  # easy / hard / elite / boss / event
        ("ascension", pa.int8()),
        ("starting_hp", pa.int16()),
        ("starting_max_hp", pa.int16()),
        # which decision
        ("decision_index", pa.int32()),
        ("turn", pa.int32()),
        ("row_kind", pa.string()),
        ("parent_action", pa.int32()),
        # encoded public state
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
        # teacher search
        ("actions", pa.list_(pa.struct([("action", pa.int32()), ("description", pa.string()),
                                        ("visits", pa.int64()), ("mean_value", F32)]))),
        ("chosen_action", pa.int32()),
        ("was_random", pa.bool_()),
        ("root_value", F32),
        ("simulations_used", pa.int64()),
        ("oracle", pa.bool_()),  # teacher searched the true state (perfect RNG foresight); NULL in older runs = false
        # this fight's outcome
        ("won", pa.bool_()),
        ("final_hp", pa.int16()),
        ("potions", pa.int8()),
        ("terminal_value", F32),
    ],
    metadata={"schema": NAME},
)
