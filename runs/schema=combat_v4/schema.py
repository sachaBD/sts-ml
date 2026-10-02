"""Schemas, file layout, DuckDB views, and generated field reference for combat_v4."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pyarrow as pa

NAME = "combat_v4"


def f(name, type, doc: str = "", nullable: bool = False):
    return pa.field(name, type, nullable=nullable, metadata={b"doc": doc.encode()} if doc else None)


def struct(*fields):
    return pa.struct(list(fields))


RNG = struct(
    f("counter", pa.int32(), "RNG call counter"),
    f("seed0", pa.uint64(), "RNG state word"),
    f("seed1", pa.uint64(), "RNG state word"),
)
CARD = struct(
    f("id", pa.int16(), "CardId"), f("unique_id", pa.int16(), "Combat-local card instance ID"),
    f("special_data", pa.int16(), "Card mutable special data"), f("cost", pa.int8(), "Base cost"),
    f("cost_for_turn", pa.int8(), "Current-turn cost"), f("upgraded", pa.bool_()),
    f("free_to_play_once", pa.bool_()), f("retain", pa.bool_()),
)
STATUS = struct(f("status", pa.int16(), "PlayerStatus"), f("value", pa.int16()), f("just_applied", pa.bool_()))
MONSTER_STATUS = struct(f("status", pa.int16(), "MonsterStatus"), f("value", pa.int32()),
                        f("just_applied", pa.bool_()))
MONSTER = struct(
    f("index", pa.int32(), "Simulator monster-slot index"), f("id", pa.int32(), "MonsterId"),
    f("hp", pa.int32()), f("max_hp", pa.int32()), f("block", pa.int32()),
    f("is_escaping", pa.bool_()), f("half_dead", pa.bool_()), f("escape_next", pa.bool_()),
    f("move_history", pa.list_(pa.int16()), "Exactly two MonsterMoveId values"),
    f("statuses", pa.list_(MONSTER_STATUS), "Active/nonzero monster statuses"),
    f("unique_power_0", pa.int32()), f("unique_power_1", pa.int16()), f("misc_info", pa.int32()),
)
PLAYER = struct(
    f("character", pa.int8(), "CharacterClass"), f("gold", pa.int32()), f("hp", pa.int32()),
    f("max_hp", pa.int32()), f("energy", pa.int32()), f("block", pa.int32()),
    f("energy_per_turn", pa.int8()), f("card_draw_per_turn", pa.int8()), f("stance", pa.int8(), "Stance"),
    f("orb_slots", pa.int8()),
    f("orbs", pa.list_(struct(f("orb", pa.int8(), "Orb"), f("data", pa.int16()))), "One entry per orb slot"),
    f("lightning_orbs_channeled", pa.int16()), f("frost_orbs_channeled", pa.int16()),
    f("last_targeted_monster", pa.int8()), f("artifact", pa.int32()), f("dexterity", pa.int32()),
    f("focus", pa.int32()), f("strength", pa.int32()),
    f("statuses", pa.list_(STATUS), "Active/nonzero player statuses"),
    f("relics", pa.list_(pa.int16()), "Active RelicId values"),
    f("happy_flower_counter", pa.int8()), f("incense_burner_counter", pa.int8()),
    f("ink_bottle_counter", pa.int8()), f("inserter_counter", pa.int8()), f("nunchaku_counter", pa.int8()),
    f("pen_nib_counter", pa.int8()), f("sundial_counter", pa.int8()),
    f("used_necronomicon_this_turn", pa.bool_()), f("combust_hp_loss", pa.int8()),
    f("deva_form_energy_per_turn", pa.int16()), f("echo_form_cards_doubled", pa.int8()),
    f("panache_counter", pa.int8()), f("cards_played_this_turn", pa.int16()),
    f("attacks_played_this_turn", pa.int16()), f("skills_played_this_turn", pa.int16()),
    f("cards_discarded_this_turn", pa.int16()), f("orange_pellets_card_types_played", pa.uint8()),
    f("last_attack_unblocked_damage", pa.int16()), f("times_damaged_this_combat", pa.int16()),
    f("bomb1", pa.int8()), f("bomb2", pa.int8()), f("bomb3", pa.int8()), f("self_repair_heal", pa.int16()),
    f("creative_ai", pa.int8()), f("hello_world", pa.int8()), f("heatsinks", pa.int8()),
    f("nirvana_block", pa.int8()), f("rushdown_draw", pa.int8()), f("study_insights", pa.int8()),
    f("mental_fortress_block", pa.int8()), f("machine_learning", pa.int8()), f("storm", pa.int8()),
)
INITIAL_STATE = struct(
    f("extra_state_json", pa.string(), "Exact scalar bits/maps and derived counters not represented by typed fields; empty queues required"),
    f("seed", pa.uint64(), "Battle seed"), f("floor_num", pa.int32()), f("encounter", pa.int16(), "MonsterEncounter"),
    f("ascension", pa.int8()), f("turn", pa.int32()), f("loop_count", pa.int32()),
    f("energy_wasted", pa.int32()), f("cards_drawn", pa.int32()), f("have_used_discovery_action", pa.bool_()),
    f("undefined_behavior_evoked", pa.bool_()), f("unsupported_effect_kind", pa.int8(), "UnsupportedEffectKind"),
    f("unsupported_effect_id", pa.int32()), f("monster_turn_idx", pa.int32()), f("is_battle_over", pa.bool_()),
    f("escaped_combat", pa.bool_()), f("end_turn_queued", pa.bool_()), f("end_turn_after_current_card", pa.bool_()),
    f("turn_has_ended", pa.bool_()), f("skip_monster_turn", pa.bool_()), f("misc_bits", pa.uint32()),
    f("rng", struct(f("ai", RNG), f("card_random", RNG), f("misc", RNG), f("monster_hp", RNG),
                    f("potion", RNG), f("shuffle", RNG))),
    f("player", PLAYER),
    f("monsters", struct(f("monsters_alive", pa.int32()), f("monster_count", pa.int32()),
                         f("extra_roll_move_on_turn", pa.uint8(), "Five-bit flag set"),
                         f("skip_turn", pa.uint8(), "Five-bit flag set"),
                         f("slots", pa.list_(MONSTER), "Exactly five simulator slots"))),
    f("cards", struct(f("next_unique_card_id", pa.int32()), f("hand", pa.list_(CARD)), f("draw", pa.list_(CARD)),
                      f("discard", pa.list_(CARD)), f("exhaust", pa.list_(CARD)),
                      f("stasis", pa.list_(CARD), "Exactly two stasis slots"))),
    f("nightmare_cards", pa.list_(CARD)), f("nightmare_copy_counts", pa.list_(pa.int32())),
    f("potions", struct(f("count", pa.int32()), f("capacity", pa.int32()),
                        f("slots", pa.list_(pa.int16()), "Exactly five Potion slots, including empty slots"))),
)


@dataclass(frozen=True)
class Table:
    name: str
    file_prefix: str
    schema: pa.Schema
    meaning: str

    def glob(self, root: Path) -> str:
        return str(root / f"schema={NAME}" / "*" / "*" / "out" / f"{self.file_prefix}-*.parquet")


TABLES = {
    "fights": Table("fights", "fights", pa.schema([
        f("fight_id", pa.string(), "Opaque globally unique combat-branch ID"),
        f("schema_version", pa.int16(), "Initial-state layout version; initially 1"),
        f("initial_state", INITIAL_STATE, "Complete initial simulator state"),
    ], metadata={b"schema": NAME.encode(), b"table": b"fights"}),
        "One complete initial simulator state per combat branch"),
    "steps": Table("steps", "steps", pa.schema([
        f("fight_id", pa.string(), "References fights.fight_id"),
        f("step_index", pa.int32(), "Zero-based action order within the fight"),
        f("action_bits", pa.uint32(), "Exact sts::search::Action::bits value executed"),
    ], metadata={b"schema": NAME.encode(), b"table": b"steps"}),
        "One executed simulator action per combat branch step"),
}


# Outcomes and search are annotations linked to the replay facts, never substituted for them.
from environments.combat.schema import COMBAT_V3
TABLES.update({
    "results": Table("results", "results", pa.schema([
        f("fight_id", pa.string()), f("run_key", pa.string()), f("run_seed", pa.uint64()),
        f("fight_index", pa.int32()), f("category", pa.string()), f("encounter", pa.string()),
        f("status", pa.string(), "completed, capped or unsupported; not inferred losses"),
        f("won", pa.bool_()), f("battle_final_hp", pa.int32()), f("battle_potions", pa.int32()),
        f("post_state_json", pa.string(), "Persistent macro state after exitBattle, before rewards"),
        f("combat_leaf", pa.string()), f("simulations", pa.int64()), f("exploration_enabled", pa.bool_()),
        f("replay_verified", pa.bool_(), "Replay matched recorded final simulator state"),
        f("replay_error", pa.string(), "Unsupported initial boundary: result retained but no canonical fight/steps or training cache"),
        f("rollout_fallback_used", pa.bool_(), "Deterministic 5k guided-rollout rescue for neural combat at turn >=30", nullable=True),
    ], metadata={b"schema": NAME.encode(), b"table": b"results"}), "Observed combat results and collection context"),
    "search": Table("search", "search", pa.schema([
        f("fight_id", pa.string()), f("step_index", pa.int32()),
        f("root_value", pa.float64()), f("simulations_used", pa.int64()),
        f("explored", pa.bool_()), f("actions_json", pa.string(), "Search visit/value annotations, not observed outcomes"),
    ], metadata={b"schema": NAME.encode(), b"table": b"search"}), "Agent search annotations at played boundaries"),
    "training": Table("training", "training", pa.schema([
        *COMBAT_V3, f("fight_id", pa.string()),
    ], metadata={b"schema": NAME.encode(), b"table": b"training", b"derived": b"encoding-cache-v1"}),
        "Disposable derived legacy-compatible encoding cache; not canonical simulator state"),
})

def table_glob(root: Path, table: str) -> str:
    return TABLES[table].glob(root)


def register_duckdb_views(db, root: Path) -> None:
    """Register combat_v4_fights and combat_v4_steps when their files exist."""
    for name, table in TABLES.items():
        if any(root.glob(f"schema={NAME}/*/*/out/{table.file_prefix}-*.parquet")):
            db.execute(f"""create view {NAME}_{name} as
                select *, concat_ws('/', schema, date, id) as run_id
                from read_parquet('{table.glob(root)}', hive_partitioning = true,
                                  hive_types_autocast = false, union_by_name = true)""")


