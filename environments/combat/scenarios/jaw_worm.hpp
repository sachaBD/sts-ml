#pragma once
// Test fixture: a seeded act-1 Ironclad fight against Jaw Worm, at its start.
#include "combat/BattleContext.h"
#include <cstdint>

namespace stsrl::scenarios {
[[nodiscard]] sts::BattleContext jaw_worm(std::uint64_t seed);
}  // namespace stsrl::scenarios
