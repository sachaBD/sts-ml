#include "agents/combat/search/config.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include <cmath>
#include <stdexcept>

namespace stsrl::search {
bool SearchConfig::parse(std::string_view flag, const std::string& value) {
    const auto check = [&](bool ok) { if (!ok) throw std::invalid_argument{"bad " + std::string{flag} + " " + value}; };
    if (flag == "--sims") check((simulations_ = std::stoi(value)) >= 0);
    else if (flag == "--particles") check((particles_ = std::stoi(value)) >= 1);
    else if (flag == "--gumbel-scale") check((gumbel_scale_ = std::stod(value)) >= 0);
    else if (flag == "--value-scale") check(std::isfinite(value_scale_ = std::stod(value)) && value_scale_ >= 0);
    else if (flag == "--rollout-mix") check((rollout_mix_ = std::stod(value)) >= 0 && rollout_mix_ <= 1);
    else return false;
    return true;
}

SearchFactory SearchConfig::factory() const {
    Gumbel::Settings gumbel;
    gumbel.simulations = simulations_;
    gumbel.particles = particles_;
    gumbel.gumbel_scale = gumbel_scale_;
    gumbel.value_scale = value_scale_;
    return [gumbel, mix = rollout_mix_](const sts::BattleContext& state, std::mt19937_64& random) {
        std::unique_ptr<CombatSearch> search = std::make_unique<Gumbel>(state, gumbel, random());
        const auto rollout_seed = random();
        return mix > 0 ? with_rollouts(std::move(search), state, mix, rollout_seed) : std::move(search);
    };
}
}  // namespace stsrl::search
