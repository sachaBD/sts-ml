#pragma once
// A fight start: the pre-battle game fields the simulator's BattleContext::init reads, as JSON
// ({"seed", "ascension", "act", "floor", "encounter", "hp", "deck", "relics", "potions", ...}; starts.py makes them).
#include "combat/BattleContext.h"
#include <nlohmann/json.hpp>

namespace stsrl::combat {
// The battle at its start, before any action. Ironclad only; throws on a malformed start.
sts::BattleContext start_battle(const nlohmann::json& start);
}  // namespace stsrl::combat
