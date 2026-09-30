// Macro simulator helpers: an act played out of combat in sts_lightspeed, with each fight's result injected
// from outside (an outcome model) instead of played. Everything is opt-in; nothing here changes sts_lightspeed's
// default behaviour. See slop_docs/card-policy-loop/macro-simulator-scoping.md.
//
// Driving loop (see run_act1 for the reference version):
//   while undecided:
//     prepare(gc, options)                    // before EVERY out-of-combat step (idempotent)
//     if gc.screenState == BATTLE: apply_battle_result(gc, result)
//     else: take one out-of-combat action (SimpleAgent::stepOutOfCombat, GameAction::execute, ...)
//   Do NOT set gc.skipBattles: it runs afterBattle() inside enterBattle, so no result can be injected (and
//   event fights would skip their custom reward lambdas).
//
// Clone / branch: GameContext is a value type; copying it copies every RNG stream, pool, deck and the pending
// regainControlAction (its lambdas take the context as an argument and capture only values). The only shared
// state is `map` (std::shared_ptr<Map>): read-only within an act, but transitionToAct overwrites *map in place,
// so a copy that crosses into act 2 changes the map of every copy sharing it. Use detach_map(gc) before that.
// SimpleAgent holds its planned map path (mapPath): copy the agent along with the context.
#pragma once

#include "game/GameContext.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace stsrl::macro_sim {

// ---------------------------------------------------------------- A. map
// Rows are map rows 0-14 (act 1: floor = row + 1); the boss (row 15) is not part of a path.
struct Path {
    std::array<sts::Room, 15> rooms{};  // Room::NONE for rows at or behind the current node
    std::vector<int> first_xs;          // x of the next node for every route giving this room sequence (ascending)
};

struct MapState {
    int y = -1;  // current row; -1 = before the first floor (at the start)
    int x = -1;
    std::vector<Path> paths;  // distinct remaining room sequences to the boss (sorted); empty once at row 14
};

MapState map_state(const sts::GameContext& gc);

// {current: {y, x, floor}, burning_elite: {x, y}, nodes: [{x, y, room, edges}], paths: [{rooms, first_xs}]}.
// nodes: rows 0-14, nodes with a room only. room: sts::roomStrings name. paths.rooms: 15 map symbols
// (N none, M monster, ? event, E elite, R rest, $ shop, T treasure).
nlohmann::json map_json(const sts::GameContext& gc);

// stsrl::game_state::state(gc) plus "map": map_json(gc).
nlohmann::json state_json(const sts::GameContext& gc);

// Give gc its own copy of the map (before crossing into another act in a branched copy).
void detach_map(sts::GameContext& gc);

// ---------------------------------------------------------------- B. common random numbers
// Opt-in. Called at a MAP_SCREEN, reseeds the run-long, decision-dependent streams from (gc.seed, next floor):
//   cardRng, potionRng, relicRng, eventRng, merchantRng, treasureRng, monsterRng.
// The per-floor streams (miscRng, shuffleRng, cardRandomRng) are already reseeded from seed + floor by the game;
// aiRng / monsterHpRng / mathUtilRng are combat-only; neowRng is used before floor 1 only.
// Idempotent within a map screen (same floor -> same seeds). No-op off the map screen.
// Diverges regardless (state, not RNG): pool contents consumed by earlier choices (relic pools pop from the
// front, events already seen, colorless pool), monster/elite list offsets (fights so far on this path),
// cardRarityFactor and potionChance (carry the outcomes of earlier rolls), deck/gold/HP-gated content
// (bottled relics, Divine Fountain / Moai Head / Cleric / Old Beggar / Woman in Blue, event outcome gates),
// the number of draws inside a floor when a choice changes it (e.g. a shop purchase, Question Card offers).
// Different path -> different rooms, of course. ? room outcomes use eventRng and the running monster/shop/
// treasure chances, which depend on the ? rooms visited so far.
void crn_reseed(sts::GameContext& gc);

// ---------------------------------------------------------------- C. fight result injection
struct BattleResult {
    bool won = true;
    int hp = 0;              // HP at the end of the fight, BEFORE end-of-combat relic heals (Burning Blood ...)
    int max_hp_delta = 0;    // e.g. Feed
    int gold_delta = 0;      // e.g. Hand of Greed; stolen gold is not modelled
    std::vector<int> potions_used;  // potion slot indices used / emptied
};

// Requires gc.screenState == BATTLE (the fight has been set up by the map / event). Applies the result as
// BattleContext::exitBattle would: HP, max HP, gold, potions; on a win Burning Blood (+6), Black Blood (+12) and
// Meat on the Bone (+12 if HP <= max / 2), in relic order, then regainControl() (afterBattle: rewards, or the
// event's custom continuation). On a loss sets outcome = PLAYER_LOSS. Not modelled: persistent relic counters
// (Pen Nib, Nunchaku, ...: left as they were), Self Repair, Ritual Dagger / Genetic Algorithm misc, Parasite
// from Writhing Mass, stolen gold, Lizard Tail / Fairy (the model's HP already includes them).
void apply_battle_result(sts::GameContext& gc, const BattleResult& result);

// ---------------------------------------------------------------- D. content filter
// Opt-in: removes content we cannot trust from the pools it spawns from. Idempotent; call it after construction
// and after every act transition (prepare() does both).
//   relics: Prayer Wheel, Lizard Tail, Prismatic Shard (plus Necronomicon for safety; only Cursed Tome gives it)
//   events: Dead Adventurer (act 1; option indices todo, relic reward drawn pre-fight), We Meet Again (bottled-card
//           todo), Falling (act 3, todo untested), Cursed Tome (act 2; Necronomicon todo)
// Already never spawns in sts_lightspeed: Smoke Bomb (rerolled in getRandomPotion). Lesson Learned is a Watcher
// card: it can only reach an Ironclad through Prismatic Shard (now removed). Colosseum stays (act 2).
void filter_content(sts::GameContext& gc);
bool is_filtered_relic(sts::RelicId relic);
bool is_filtered_event(sts::Event event);

// ---------------------------------------------------------------- driving
struct Options {
    bool crn = false;
    bool filter = false;
};

// Call before every out-of-combat step: filter_content if options.filter, crn_reseed if options.crn.
void prepare(sts::GameContext& gc, const Options& options);

// Reference loop: SimpleAgent plays out of combat; `fight` returns each fight's result. Returns once the act 1
// boss is beaten (true) or the run is lost (false).
using FightFn = std::function<BattleResult(const sts::GameContext& gc)>;
bool run_act1(sts::GameContext& gc, const Options& options, const FightFn& fight);

}  // namespace stsrl::macro_sim
