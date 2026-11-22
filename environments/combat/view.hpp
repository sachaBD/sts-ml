#pragma once
#include "combat/BattleContext.h"
#include <nlohmann/json.hpp>

namespace stsrl::combat {
// What the player sees: the draw pile as a sorted multiset, never its order or the RNG.
// `legal` uses the same deduplicated moves and semantic action keys as search.
nlohmann::json view(const sts::BattleContext&);
}  // namespace stsrl::combat
