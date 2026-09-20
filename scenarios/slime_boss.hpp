#pragma once

#include "combat/environment.hpp"

#include <cstdint>

namespace stsrl::scenarios {

[[nodiscard]] CombatEnvironment slime_boss(std::uint64_t seed);
[[nodiscard]] CombatEnvironment slime_boss_after_split(std::uint64_t seed, int hp);

} // namespace stsrl::scenarios
