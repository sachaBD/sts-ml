#include "agents/combat/search/config.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include <cmath>
#include <stdexcept>

namespace stsrl::search {
void SearchConfig::validate() const {
    if (sims < 0 || particles < 1 || !std::isfinite(gumbel_scale) || gumbel_scale < 0 ||
        !(rollout_mix >= 0 && rollout_mix <= 1))
        throw std::invalid_argument{"search: nonnegative sims and gumbel scale, positive particles, rollout mix in [0, 1]"};
}

SearchFactory SearchConfig::factory() const {
    validate();
    Gumbel::Settings gumbel;
    gumbel.simulations = sims;
    gumbel.particles = particles;
    gumbel.gumbel_scale = gumbel_scale;
    return [gumbel, mix = rollout_mix](const sts::BattleContext& state, std::mt19937_64& random) {
        std::unique_ptr<CombatSearch> search = std::make_unique<Gumbel>(state, gumbel, random());
        const auto rollout_seed = random();
        return mix > 0 ? with_rollouts(std::move(search), state, mix, rollout_seed) : std::move(search);
    };
}
}  // namespace stsrl::search
