#pragma once
// The one place that knows which search runs and how it is configured. Callers see only a SearchFactory.
#include "agents/combat/search/search.hpp"

namespace stsrl::search {
struct SearchConfig {
    int simulations, particles;
    double gumbel_scale;  // 1 explores (self-play); 0 is deterministic (evaluation)
    double rollout_mix;   // above 0, each leaf value is blended with a guided rollout's outcome

    // Gumbel search; throws on settings out of range.
    SearchFactory factory() const;
};
}  // namespace stsrl::search
