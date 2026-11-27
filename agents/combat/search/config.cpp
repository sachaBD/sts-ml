#include "agents/combat/search/config.hpp"
#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include <cmath>
#include <stdexcept>

namespace stsrl::search {
void SearchConfig::validate() const {
    if (sims < 0 || particles < 1 || considered_actions < 1 || !std::isfinite(gumbel_scale) || gumbel_scale < 0 ||
        !(rollout_mix >= 0 && rollout_mix <= 1) || !std::isfinite(sieve_gain) || sieve_gain < 0)
        throw std::invalid_argument{"search: nonnegative sims, gumbel scale and sieve gain, positive particles and considered actions, rollout mix in [0, 1]"};
    if (algorithm != "gumbel" && algorithm != "horizon" && algorithm != "unified")
        throw std::invalid_argument{"search: unknown algorithm " + algorithm};
    if (algorithm == "unified" && (gumbel_scale != 0 || rollout_mix != 0))
        throw std::invalid_argument{"unified: deterministic play only; set gumbel_scale and rollout_mix to zero"};
    if (horizon.init_contenders < 2 || horizon.init_worlds < 1 || horizon.wave_worlds < 1 ||
        horizon.max_plunge_choices < 1 || horizon.horizon_turns < 1 ||
        !std::isfinite(horizon.c_puct) || horizon.c_puct < 0 ||
        !std::isfinite(horizon.readout_k0) || horizon.readout_k0 <= 0 ||
        !std::isfinite(horizon.prior_se) || horizon.prior_se < 0 ||
        !std::isfinite(horizon.lambda) || horizon.lambda < 0)
        throw std::invalid_argument{"search: invalid horizon settings"};
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
    settings.record_worlds = record_worlds;
    settings.sieve_gain = sieve_gain;
    auto gumbel = std::make_unique<Gumbel>(state, settings, random());
    auto& inspection = *gumbel;
    std::unique_ptr<CombatSearch> search = std::move(gumbel);
    const auto rollout_seed = random();
    if (rollout_mix > 0) search = with_rollouts(std::move(search), state, rollout_mix, rollout_seed);
    return {std::move(search), inspection};
}

SearchFactory SearchConfig::factory() const {
    validate();
    if (algorithm == "horizon" || algorithm == "unified") {
        return [config = *this](const sts::BattleContext& state, int simulations, std::mt19937_64& random)
            -> std::unique_ptr<CombatSearch> {
            auto hs = config.horizon;
            hs.use_thinking = config.algorithm == "unified";
            hs.simulations = simulations;
            hs.particles = config.particles;
            hs.world_draw = config.world_draw;
            return std::make_unique<Horizon>(state, hs, random());
        };
    }
    return [config = *this](const sts::BattleContext& state, int simulations, std::mt19937_64& random) {
        return config.create(state, simulations, random).search;
    };
}
}  // namespace stsrl::search
