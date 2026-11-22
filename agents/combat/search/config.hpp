#pragma once
// The one place that constructs the configured search, including rollout blending.
#include "agents/combat/search/gumbel/gumbel.hpp"

namespace stsrl::search {
struct GumbelSearch {
    std::unique_ptr<CombatSearch> search;
    Gumbel& gumbel;  // inspection and steering handle; owned by search, even when wrapped for rollouts
};

struct SearchConfig {
    int sims, particles;  // sims: the full budget
    double gumbel_scale;  // 1 explores (self-play); 0 is deterministic (evaluation)
    double rollout_mix;   // above 0, each leaf value is blended with a guided rollout's outcome
    int considered_actions = Gumbel::Settings{}.considered_actions;  // root sequential-halving candidates
    std::uint64_t world_draw = Gumbel::Settings{}.world_draw;
    bool record_worlds = Gumbel::Settings{}.record_worlds;  // per-link world visits for inspection only

    void validate() const;  // throws on settings out of range
    GumbelSearch create(const sts::BattleContext&, int simulations, std::mt19937_64&) const;  // validates first
    // Gumbel search; validates first.
    SearchFactory factory() const;
};
}  // namespace stsrl::search
