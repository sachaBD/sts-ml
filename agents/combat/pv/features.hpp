#pragma once
// PV inputs adapt the existing combat encoding; no duplicate card-mechanics implementation.
// Contract pv_champ_hp_v2: context[65], cards[N,18], monsters[N,29], potions[N,19], relics[N,4], actions[N,261].
// Training replays combat_v4 through this same encoder used by search. Zero-ID rows pad sets.
#include <array>
#include <span>
#include <vector>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"

namespace stsrl::pv {
inline constexpr const char* contract = "pv_champ_hp_v2";
inline constexpr std::array<const char*, 6> names{"context", "cards", "monsters", "potions", "relics", "actions"};
inline constexpr std::array<int, 6> widths{65, 18, 29, 19, 4, 261};
using Inputs = std::array<std::vector<float>, 6>;  // flattened rows; context has exactly one row
Inputs encode(const sts::BattleContext&, std::span<const sts::search::Action>);
}  // namespace stsrl::pv
