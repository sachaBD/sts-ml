#pragma once
#include "combat/BattleContext.h"
#include <nlohmann/json.hpp>

namespace stsrl::combat {
// What the player sees: the draw pile as a sorted multiset, never its order or the RNG.
// `legal` lists every legal move, one per hand slot.
nlohmann::json view(const sts::BattleContext&);
}  // namespace stsrl::combat
