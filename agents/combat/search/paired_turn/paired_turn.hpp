#pragma once
#include "agents/combat/search/search.hpp"
#include <cstdint>

namespace stsrl::search {
// Compare moves on shared sampled worlds after a greedy continuation to the next turn.
SearchFactory paired_turn(int particles, std::uint64_t world_draw);
}  // namespace stsrl::search
