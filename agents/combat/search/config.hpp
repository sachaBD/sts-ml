#pragma once
// The one place that constructs the configured search, including rollout blending.
#include "agents/combat/search/gumbel/gumbel.hpp"
#include <memory>
#include <random>

namespace stsrl::search {
struct GumbelSearch {
    std::unique_ptr<CombatSearch> search;
    const Gumbel& gumbel;  // inspection view; owned by search, even when wrapped for rollouts
};

struct SearchConfig {
    int sims, particles;
    double gumbel_scale;  // 1 explores (self-play); 0 is deterministic (evaluation)
    double rollout_mix;   // above 0, each leaf value is blended with a guided rollout's outcome

    void validate() const;  // throws on settings out of range
    // A fresh search for one decision, drawing all its randomness from the generator; validates first.
    GumbelSearch create(const sts::BattleContext&, std::mt19937_64&) const;
};
}  // namespace stsrl::search
