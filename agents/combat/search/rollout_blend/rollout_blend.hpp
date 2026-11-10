#pragma once
// AlphaGo-style leaf value: (1 - mix) * network + mix * the outcome of a guided rollout from the query's state.
#include "agents/combat/search/search.hpp"
#include <cstdint>
#include <memory>

namespace stsrl::search {
// Wraps any search so every evaluation it receives is blended with a rollout from that query's state.
std::unique_ptr<CombatSearch> with_rollouts(std::unique_ptr<CombatSearch>, const sts::BattleContext& any_state,
                                            double mix, std::uint64_t seed);
}  // namespace stsrl::search
