#pragma once
#include "agents/combat/pv/turn_search.hpp"
#include <map>

namespace stsrl::pv {
struct ParticleActionValues {
    double root_value = 0;
    std::map<std::uint64_t, double> best;
};
// Missing actions use the particle's clamped network root value, never zero or max-child V.
std::vector<double> mean_particle_scores(std::span<const std::uint64_t> keys,
                                         std::span<const ParticleActionValues> particles);
ParticleActionValues turn_first_action_values(const sts::BattleContext&, const TurnNode&, double root_value);
struct TurnPimcResult {
    sts::search::Action action;
    std::vector<double> scores;
    double seconds = 0, mean_depth = 0;
    std::size_t particles = 0, fallback_count = 0, time_fallback_count = 0, time_overshoot_count = 0;
    std::size_t network_calls = 0, evaluated_states = 0;
    std::vector<std::string> fallback_reasons;
};
// Independent, sequential determinized searches; no cross-decision or cross-particle reuse.
TurnPimcResult decide_turn_particles(const sts::BattleContext& observed,
    std::span<const sts::BattleContext> particles, std::size_t budget,
    const TurnSearch::Evaluate&, TurnSearchCaps = {}, double c = 1.25);
} // namespace stsrl::pv
