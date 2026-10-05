"""megacrit_champ_v1: reconstructed state at the START of the Champ fight for human StS1 runs (megacrit_runs_v1).

Producer: apps/megacrit_dump/champ_starts.py. Method: start from the end-of-run record (master_deck, relics) and undo,
in descending floor order, every change made on/after the Champ floor F (card rewards, shop buys/purges, events,
smiths after F, relics, boss relics after the Champ). Undos that cannot be done cleanly are not guessed: they add an
`issues` entry and set exact = false. Potions are not reconstructed (not reliably recoverable) and are not stored.
Layout: out/champ_starts.parquet (one row per Champ fight, zstd).
"""
import pyarrow as pa

NAME = "megacrit_champ_v1"

DECK_CARD = pa.struct([("card", pa.string()), ("upgrades", pa.int32())])

CHAMP_STARTS = pa.schema([
    ("play_id", pa.string()),
    ("source_run_id", pa.string()),            # megacrit_runs_v1 run id (schema/date/id) the record came from
    ("build_version", pa.string()),
    ("timestamp", pa.int64()),
    ("F", pa.int32()),                         # floor of the Champ fight
    ("champ_won", pa.bool_()),                 # floor_reached > F or victory
    ("champ_damage", pa.int32()),
    ("champ_turns", pa.int32()),
    ("hp", pa.int32()),                        # current_hp_per_floor[F-2]; null if unavailable
    ("max_hp", pa.int32()),                    # max_hp_per_floor[F-2]
    ("deck_raw", pa.list_(pa.string())),       # dump card strings (with "+N"), sorted
    ("deck", pa.list_(DECK_CARD)),             # sts_lightspeed CardId enum name lowercased + upgrade count; sorted
    ("relics_raw", pa.list_(pa.string())),     # dump relic strings, sorted
    ("relics", pa.list_(pa.string())),         # sts_lightspeed RelicId enum names lowercased, sorted
    ("unmapped", pa.list_(pa.string())),       # card/relic strings with no sts_lightspeed id (omitted from deck/relics)
    ("exact", pa.bool_()),                     # true iff issues is empty
    ("issues", pa.list_(pa.string())),         # "<reason>:<detail>"
])
