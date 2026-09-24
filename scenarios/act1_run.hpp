// One seeded Ironclad act 1: SimpleAgent plays everything out of combat, the caller plays each combat.
// Shared by apps/bootstrap (teacher plays every fight) and apps/value_play (replays stored fights).
#pragma once

#include "combat/BattleContext.h"
#include "game/GameContext.h"

#include <cstdint>
#include <functional>
#include <string>

#include <nlohmann/json.hpp>

namespace stsrl::act1 {

// Plays one combat from its start state; `fight` = the combat_v3 columns shared by its rows.
// Returns the finished battle.
using FightFn = std::function<sts::BattleContext(const sts::BattleContext& start, const nlohmann::json& fight)>;

struct Result {
    std::string status;  // died, act_complete, or stopped (max_fights reached)
    int floor = 0;
    int fights = 0;
};

// Plays until death, the act 1 boss is beaten, or `max_fights` combats have been played.
Result play(sts::GameContext& game, const FightFn& fight, int max_fights = -1);

}  // namespace stsrl::act1
