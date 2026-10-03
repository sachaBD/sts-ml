"""combat_v4_full: every decision of a combat_v4 fight, expanded. Documentation as code.

Derived only: the decompressor rebuilds each combat_v4 fight (runs/schema=combat_v4/schema.py) with sts_lightspeed
and writes one `decisions` row per decision point plus one `fights` row per fight. Nothing here is stored
canonically; it can be deleted and regenerated at the same commits. It may be materialized (an analysis or
training run), or produced on the fly.

Ironclad only, explicit only:
- Statuses are generic (name, amount) lists: every PlayerStatus / MonsterStatus is representable.
- Fields for other characters do not exist. The decompressor crashes if one is non-default (orb slots, orbs,
  stance, focus, mantra, Defect / Watcher counters).
- Raw per-monster integers (Monster::miscInfo, uniquePower0, uniquePower1) are decoded into named `counters` by a
  per-MonsterId table in the decompressor. A non-zero raw value without a decoding for that monster: crash.
- A decision boundary the simulator cannot enumerate (Action::getAllActionsInState is empty while the fight is
  undecided): crash.

Search fields (legal[].visits / value, root_value, simulations) come from the combat_v4 `search` rows of the
playing agent (fights.agent), joined on (fight_id, step); null where it has none. `explored` comes from
fights.explored.

Public vs hidden: `public` is what a human player sees on screen (draw pile as a multiset, intents unless Runic
Dome). `hidden` is the rest of the true state (draw order, RNG streams, intents hidden by Runic Dome). Models must
read `public` only; `hidden` exists for analysis, oracles and debugging.

Names are lower-case sts_lightspeed enum names (cards: cardEnumStrings, relics: relicEnumNames, potions:
potionEnumNames, monsters: monsterIdStrings, moves: monster move enum, statuses: playerStatusEnumStrings /
MonsterStatus names, encounters: monsterEncounterEnumNames). Dictionary-encoded on write.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa

NAME = "combat_v4_full"
SOURCE = "combat_v4"


def f(name, type, doc, nullable=False):
    return pa.field(name, type, nullable=nullable, metadata={b"doc": doc.encode()})


STATUS = pa.struct([f("status", pa.string(), "status name"), f("amount", pa.int32(), "stacks / turns; 1 for flags")])

CARD = pa.struct([
    f("card", pa.string(), "card name"),
    f("upgrades", pa.int8(), "0 / 1 (Searing Blow: count)"),
    f("cost", pa.int8(), "CardInstance::cost (-1 unplayable, -2 X)"),
    f("cost_for_turn", pa.int8(), "CardInstance::costForTurn: what playing it now costs"),
    f("special_data", pa.int16(), "CardInstance::specialData (e.g. Rampage / Ritual Dagger / Genetic Algorithm value)"),
    f("free_to_play_once", pa.bool_(), "CardInstance::freeToPlayOnce"),
    f("retain", pa.bool_(), "CardInstance::retain"),
    f("unique_id", pa.int16(), "combat-local card instance id (links hand / piles / hidden.draw_order)"),
])

RELIC = pa.struct([
    f("relic", pa.string(), "relic name"),
    f("counter", pa.int16(), "in-combat counter where the simulator keeps one (Pen Nib, Nunchaku, Ink Bottle, "
                             "Happy Flower, Incense Burner, Sundial), else null", nullable=True),
])

MONSTER = pa.struct([
    f("slot", pa.int8(), "simulator monster slot 0-4 (targets refer to it)"),
    f("monster", pa.string(), "MonsterId name"),
    f("alive", pa.bool_(), "targetable and not dead / escaped"),
    f("hp", pa.int16(), "current HP"),
    f("max_hp", pa.int16(), "max HP"),
    f("block", pa.int16(), "block"),
    f("intent", pa.string(), "current move (moveHistory[0]); null when hidden by Runic Dome", nullable=True),
    f("previous_move", pa.string(), "moveHistory[1]", nullable=True),
    f("intent_damage", pa.int16(), "per-hit damage of the intent as shown (after strength / weak / vulnerable)",
      nullable=True),
    f("intent_hits", pa.int8(), "hit count of the intent", nullable=True),
    f("statuses", pa.list_(STATUS), "every non-zero MonsterStatus (strength, artifact, ..., flags as 1)"),
    f("counters", pa.list_(STATUS), "decoded per-monster raw state, e.g. champ phase, hexaghost orbs, wizard charge"),
    f("half_dead", pa.bool_(), "Darkling / Awakened One between phases"),
    f("escaping", pa.bool_(), "is escaping / escapes next turn"),
    f("stasis", CARD, "card held by this Bronze Orb (Stasis), else null", nullable=True),
])

PLAYER = pa.struct([
    f("hp", pa.int16(), ""), f("max_hp", pa.int16(), ""), f("block", pa.int16(), ""),
    f("energy", pa.int8(), "current energy"), f("energy_per_turn", pa.int8(), ""),
    f("draw_per_turn", pa.int8(), "cards drawn at turn start"), f("gold", pa.int16(), ""),
    f("statuses", pa.list_(STATUS), "every non-zero PlayerStatus, incl. strength / dexterity / artifact"),
    f("cards_played_this_turn", pa.int8(), ""), f("attacks_played_this_turn", pa.int8(), ""),
    f("skills_played_this_turn", pa.int8(), ""), f("cards_discarded_this_turn", pa.int8(), ""),
    f("orange_pellets_types", pa.uint8(), "card types played this turn, bit per CardType"),
    f("times_damaged_this_combat", pa.int16(), ""),
    f("used_necronomicon_this_turn", pa.bool_(), ""),
    f("combust_hp_loss", pa.int8(), ""),
    f("bombs", pa.list_(pa.int16()), "The Bomb damage queued for 1 / 2 / 3 turns from now, exactly 3"),
])

PUBLIC = pa.struct([
    f("turn", pa.int16(), "BattleContext::turn"),
    f("player", PLAYER, ""),
    f("relics", pa.list_(RELIC), "in relic order"),
    f("potions", pa.list_(pa.string()), "one per slot (capacity entries), null = empty slot"),
    f("hand", pa.list_(CARD), "hand in order (card actions index it)"),
    f("draw", pa.list_(CARD), "draw pile as a multiset: sorted by card, upgrades; order is hidden"),
    f("discard", pa.list_(CARD), "discard pile, oldest first"),
    f("exhaust", pa.list_(CARD), "exhaust pile, oldest first"),
    f("monsters", pa.list_(MONSTER), "occupied slots in slot order"),
])

HIDDEN = pa.struct([
    f("draw_order", pa.list_(pa.int16()), "unique_id of draw-pile cards, next draw first"),
    f("rng", pa.list_(pa.struct([f("stream", pa.string(), "ai, card_random, misc, monster_hp, potion, shuffle"),
                                 f("counter", pa.int32(), ""), f("seed0", pa.uint64(), ""),
                                 f("seed1", pa.uint64(), "")])), "battle RNG streams"),
    f("hidden_intents", pa.list_(pa.string()), "per entry of public.monsters: intent when Runic Dome hides it",
      nullable=True),
])

LEGAL = pa.struct([
    f("action", pa.uint32(), "Action::bits"),
    f("visits", pa.uint32(), "search visits (null: not searched / not visited)", nullable=True),
    f("value", pa.float32(), "search mean value (null: not searched / not visited)", nullable=True),
])

DECISIONS = pa.schema([
    f("fight_id", pa.string(), "combat_v4 fight_id"),
    f("step", pa.int16(), "index into combat_v4 actions"),
    f("kind", pa.string(), "'play' (normal turn input) or the card-select task name (exhaust, discard, ...)"),
    f("public", PUBLIC, "what the player sees before choosing"),
    f("hidden", HIDDEN, "true state beyond what the player sees"),
    f("legal", pa.list_(LEGAL), "every legal move, Action::getAllActionsInState order"),
    f("chosen", pa.int16(), "index into legal of the move played"),
    f("explored", pa.bool_(), "combat_v4 fights.explored[step]"),
    f("root_value", pa.float32(), "search root value", nullable=True),
    f("simulations", pa.uint32(), "search simulations", nullable=True),
], metadata={b"schema": NAME.encode(), b"table": b"decisions"})

FIGHTS = pa.schema([
    f("fight_id", pa.string(), "combat_v4 fight_id"),
    f("encounter", pa.string(), "encounter name"),
    f("act", pa.int8(), ""), f("floor", pa.int16(), ""), f("ascension", pa.int8(), ""),
    f("room", pa.string(), "monster / elite / boss / event"),
    f("agent", pa.string(), "combat_v4 agent"),
    f("decisions", pa.int16(), "number of decisions (= len(actions))"),
    f("turns", pa.int16(), "turns played"),
    f("won", pa.bool_(), ""),
    f("start_hp", pa.int16(), "player HP after battle start (start-of-combat heals applied)"),
    f("final_hp", pa.int16(), ""),
    f("max_hp", pa.int16(), "at the end"),
    f("potions_start", pa.int8(), ""), f("potions_end", pa.int8(), ""),
    f("monster_hp_left", pa.int16(), "sum of living monsters' HP at the end (0 on a win)"),
], metadata={b"schema": NAME.encode(), b"table": b"fights"})

TABLES = {"decisions": DECISIONS, "fights": FIGHTS}


def register_duckdb_views(db, root: Path) -> None:
    """combat_v4_full_decisions / combat_v4_full_fights, when materialized."""
    for table in TABLES:
        glob = f"schema={NAME}/*/*/out/{table}-*.parquet"
        if any(root.glob(glob)):
            db.execute(f"""create view {NAME}_{table} as
                select *, concat_ws('/', schema, date, id) as run_id
                from read_parquet('{root}/{glob}', hive_partitioning = true, hive_types_autocast = false)""")
