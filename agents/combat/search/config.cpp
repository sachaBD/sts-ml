#include "agents/combat/search/config.hpp"
#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include <cmath>
#include <stdexcept>

namespace stsrl::search {
void SearchConfig::validate() const {
    if (sims < 0 || particles < 1 || considered_actions < 1 || !std::isfinite(gumbel_scale) || gumbel_scale < 0 ||
        !(rollout_mix >= 0 && rollout_mix <= 1))
        throw std::invalid_argument{"search: nonnegative sims and gumbel scale, positive particles and considered actions, rollout mix in [0, 1]"};
}

GumbelSearch SearchConfig::create(const sts::BattleContext& state, int simulations, std::mt19937_64& random) const {
    validate();
    if (simulations < 0) throw std::invalid_argument{"search: nonnegative simulations"};
    Gumbel::Settings settings;
    settings.simulations = simulations;
    settings.particles = particles;
    settings.gumbel_scale = gumbel_scale;
    settings.considered_actions = considered_actions;
    settings.world_draw = world_draw;
    auto gumbel = std::make_unique<Gumbel>(state, settings, random());
    auto& inspection = *gumbel;
    std::unique_ptr<CombatSearch> search = std::move(gumbel);
    const auto rollout_seed = random();
    if (rollout_mix > 0) search = with_rollouts(std::move(search), state, rollout_mix, rollout_seed);
    return {std::move(search), inspection};
}

SearchFactory SearchConfig::factory() const {
    validate();
    return [config = *this](const sts::BattleContext& state, int simulations, std::mt19937_64& random) {
        return config.create(state, simulations, random).search;
    };
}
}  // namespace stsrl::search
