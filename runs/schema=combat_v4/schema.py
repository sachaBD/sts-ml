"""combat_v4: the minimal stored record of an Ironclad fight. Documentation as code.

Goal: the least data that rebuilds every decision of a fight exactly, using sts_lightspeed only (no agent code),
at the sts_combat_rl / sts_lightspeed commits recorded in the run's run.json. Everything else (per-decision states,
legal moves, descriptions, NN inputs) is derived by the decompressor into combat_v4_full
(runs/schema=combat_v4_full/schema.py).

Rebuild of one fight (pure sts_lightspeed):
  1. gc = sts::GameContext(IRONCLAD, start.seed, start.ascension); then overwrite the fields BattleContext::init
     reads (START below): act, floorNum, curRoom, lastRoom, curHp, maxHp, gold, miscRng, potionRng, potions,
     potionCapacity, potionCount, relics (RelicContainer::add, in order: no pickup effects), deck
     (Deck::obtainRaw, in order), deck.bottleIdxs, info.encounter; the burning elite iff burning_elite_buff >= 0
     (curMapNode = the map's burning elite node, map->burningEliteBuff = burning_elite_buff).
  2. bc.init(gc)  (the game's own battle start: shuffles, innates, start-of-combat relics, monster rolls).
  3. for bits in actions: sts::search::Action(bits).execute(bc).
  4. bc must have ended: outcome == PLAYER_VICTORY iff won, bc.player.curHp == final_hp. Any mismatch: crash.

Why the pre-init game fields, not a BattleContext snapshot: they are exactly what the game uses to start a fight,
they are small (the deck dominates), and every fight start is representable (a post-init snapshot cannot hold
pending callback actions, e.g. Gambling Chip / Enchiridion starts).

Scope: Ironclad only. The writer crashes on anything else (character, unsupported start). No fields exist for
other characters (orbs, stances, focus, ...).

Tables: `fights` (the replay facts, one row per fight) and `search` (agent search statistics, one row per searched
decision; optional, joined on fight_id + step). Search statistics cannot be rebuilt, so they are stored when wanted;
they never affect the rebuild.

File layout: out/<table>-<part>.parquet, many fights per file (a worker or run flushes batches; never one file
per run). zstd.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa

NAME = "combat_v4"
VERSION = 1  # start/actions layout; bump on any reinterpretation


def f(name, type, doc, nullable=False):
    return pa.field(name, type, nullable=nullable, metadata={b"doc": doc.encode()})


RNG = pa.struct([
    f("counter", pa.int32(), "sts::Random::counter"),
    f("seed0", pa.uint64(), "sts::Random::seed0"),
    f("seed1", pa.uint64(), "sts::Random::seed1"),
])

DECK_CARD = pa.struct([
    f("id", pa.int16(), "sts::CardId"),
    f("upgraded", pa.bool_(), "sts::Card::upgraded"),
    f("misc", pa.int16(), "sts::Card::misc (Searing Blow: upgrade count; Ritual Dagger / Genetic Algorithm: value)"),
])

RELIC = pa.struct([
    f("id", pa.int16(), "sts::RelicId"),
    f("data", pa.int32(), "sts::RelicInstance::data (counters such as Pen Nib, Nunchaku, Lizard Tail used)"),
])

# Every GameContext field BattleContext::init (sts_lightspeed src/combat/BattleContext.cpp, CardManager::init,
# initRelics) reads. The character is IRONCLAD and is not stored.
START = pa.struct([
    f("seed", pa.uint64(), "GameContext::seed (battle RNG streams start from Random(seed + floor))"),
    f("ascension", pa.int8(), "GameContext::ascension"),
    f("act", pa.int8(), "GameContext::act (used by the burning elite buff)"),
    f("floor", pa.int16(), "GameContext::floorNum"),
    f("encounter", pa.int8(), "sts::MonsterEncounter (GameContext::info.encounter)"),
    f("cur_room", pa.int8(), "sts::Room of the fight (GameContext::curRoom)"),
    f("last_room", pa.int8(), "sts::Room before it (GameContext::lastRoom; Ancient Tea Set)"),
    f("burning_elite_buff", pa.int8(), "-1: not the burning elite; else Map::burningEliteBuff (0-3)"),
    f("hp", pa.int16(), "GameContext::curHp before battle start (start-of-combat heals are applied by init)"),
    f("max_hp", pa.int16(), "GameContext::maxHp"),
    f("gold", pa.int16(), "GameContext::gold"),
    f("misc_rng", RNG, "GameContext::miscRng"),
    f("potion_rng", RNG, "GameContext::potionRng"),
    f("potion_capacity", pa.int8(), "GameContext::potionCapacity (2-5)"),
    f("potions", pa.list_(pa.int8()),
      "sts::Potion per slot, exactly potion_capacity entries in slot order (1 = EMPTY_POTION_SLOT)"),
    f("relics", pa.list_(RELIC), "GameContext::relics.relics in container order (order of start-of-combat effects)"),
    f("deck", pa.list_(DECK_CARD), "GameContext::deck.cards in deck order (the battle shuffle depends on it)"),
    f("bottled", pa.list_(pa.int8()),
      "Deck::bottleIdxs, exactly 3 entries indexed by CardType ATTACK, SKILL, POWER: deck index, -1 = none"),
])

CHILD = pa.struct([
    f("action", pa.uint32(), "Action::bits of a root move the search visited"),
    f("visits", pa.uint32(), "simulations through that move"),
    f("value", pa.float32(), "mean backed-up value (the search's own normalized objective)"),
])

FIGHTS = pa.schema([
    f("fight_id", pa.string(), "opaque, globally unique; overworld_v1 steps and search rows link to it"),
    f("version", pa.int8(), "VERSION of this layout"),
    f("start", START, "pre-battle game fields; rebuild step 1"),
    f("actions", pa.list_(pa.uint32()), "sts::search::Action::bits executed, in order, from init to the end"),
    f("explored", pa.list_(pa.bool_()),
      "aligned with actions: the move was an exploration move, not the playing agent's choice"),
    f("won", pa.bool_(), "replay check: the fight ended in PLAYER_VICTORY"),
    f("final_hp", pa.int16(), "replay check: BattleContext player.curHp at the end"),
    f("agent", pa.string(), "who chose the moves, e.g. 'mcts leaf=guided_rollout sims=20000 particles=8'"),
], metadata={b"schema": NAME.encode(), b"table": b"fights"})

# Agent annotations, never needed for the rebuild. Key (fight_id, step, agent): one row per decision an agent
# searched. The playing agent's rows are written with the fight when search stats are on; any other agent may
# add rows later for recorded positions (re-annotation) without touching fights. For an agent that has rows for a
# fight, a missing step is a forced move (one legal move, no search); a fight with no rows for an agent was not
# annotated by it.
SEARCH = pa.schema([
    f("fight_id", pa.string(), "fights.fight_id"),
    f("step", pa.int16(), "index into fights.actions: the decision before that action"),
    f("agent", pa.string(), "the search that produced these statistics (same format as fights.agent)"),
    f("root_value", pa.float32(), "visit-weighted mean value at the root"),
    f("simulations", pa.uint32(), "simulations run for this decision"),
    f("children", pa.list_(CHILD), "visited root moves only, in any order"),
], metadata={b"schema": NAME.encode(), b"table": b"search"})

TABLES = {"fights": FIGHTS, "search": SEARCH}


def register_duckdb_views(db, root: Path) -> None:
    """combat_v4_fights / combat_v4_search: every row plus run_id."""
    for table in TABLES:
        glob = f"schema={NAME}/*/*/out/{table}-*.parquet"
        if any(root.glob(glob)):
            db.execute(f"""create view {NAME}_{table} as
                select *, concat_ws('/', schema, date, id) as run_id
                from read_parquet('{root}/{glob}', hive_partitioning = true, hive_types_autocast = false)""")
