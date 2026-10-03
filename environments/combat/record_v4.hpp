// combat_v4 (runs/schema=combat_v4/schema.py): the fight start record and the rebuild, sts_lightspeed only.
//
//   start_json(gc)            the pre-battle GameContext fields BattleContext::init reads (call it right before
//                             battle.init(gc)); Ironclad only, throws otherwise.
//   start_game(start)         a GameContext with those fields, ready for BattleContext::init.
//   replay(start, actions)    init + Action(bits).execute for every action; the finished battle.
//                             Throws if an action is not valid or the fight does not end exactly after the last.
#pragma once

#include "combat/BattleContext.h"
#include "game/GameContext.h"

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace stsrl::combat_v4 {

constexpr int version = 1;

// The `start` struct of runs/schema=combat_v4/schema.py as plain C++.
struct RngState { std::int32_t counter; std::uint64_t seed0, seed1; };
struct StartRelic { int id; int data; };
struct StartCard { int id; bool upgraded; int misc; };
struct Start {
    std::uint64_t seed;
    int ascension, act, floor, encounter, cur_room, last_room, burning_elite_buff, hp, max_hp, gold;
    RngState misc_rng, potion_rng;
    int potion_capacity;
    std::vector<int> potions;
    std::vector<StartRelic> relics;
    std::vector<StartCard> deck;
    int bottled[3];
};

nlohmann::json start_json(const sts::GameContext& gc);
Start start_from_json(const nlohmann::json& start);
sts::GameContext start_game(const Start& start);
sts::GameContext start_game(const nlohmann::json& start);
sts::BattleContext replay(const nlohmann::json& start, const std::vector<std::uint32_t>& actions);

}  // namespace stsrl::combat_v4
