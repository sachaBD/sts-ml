#include "agents/combat/pv/turn_pimc.hpp"
#include "agents/combat/pv/search.hpp"
#include <chrono>

namespace stsrl::pv {
std::vector<double> mean_particle_scores(std::span<const std::uint64_t> keys,
                                         std::span<const ParticleActionValues> particles) {
    if (particles.empty()) throw std::invalid_argument{"turn PIMC: no particles"};
    std::vector<double> scores(keys.size(), 0);
    for (const auto& particle : particles) {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const auto found = particle.best.find(keys[i]);
            scores[i] += found == particle.best.end() ? particle.root_value : found->second;
        }
    }
    for (auto& score : scores) score /= particles.size();
    return scores;
}
ParticleActionValues turn_first_action_values(const sts::BattleContext& state, const TurnNode& root, double value) {
    ParticleActionValues result; result.root_value = value;
    for (const auto& child : root.children) {
        if (child.sequence.empty()) throw std::runtime_error{"turn PIMC: empty child sequence"};
        const auto key = action_key(state, sts::search::Action{child.sequence.front()});
        const auto [found, inserted] = result.best.emplace(key, child.q());
        if (!inserted) found->second = std::max(found->second, child.q());
    }
    return result;
}
TurnPimcResult decide_turn_particles(const sts::BattleContext& observed,
    std::span<const sts::BattleContext> particles, std::size_t budget,
    const TurnSearch::Evaluate& evaluate, TurnSearchCaps caps, double c) {
    const auto start = std::chrono::steady_clock::now();
    validate_root_state(observed);
    const auto moves = legal_actions(observed);
    TurnPimcResult result; result.particles = particles.size();
    std::vector<std::uint64_t> keys;
    for (auto move : moves) keys.push_back(action_key(observed, move));
    std::vector<ParticleActionValues> values;
    auto counted = [&](std::span<const Inputs> batch) {
        ++result.network_calls; result.evaluated_states += batch.size();
        return evaluate(batch);
    };
    for (const auto& particle : particles) {
        validate_root_state(particle);
        if (observation_key(particle) != observation_key(observed))
            throw std::runtime_error{"turn PIMC: particle changed public observation"};
        const auto particle_moves = legal_actions(particle);
        const std::array<Inputs, 1> input{encode(particle, particle_moves)};
        auto predictions = counted(input);
        if (predictions.size() != 1) throw std::runtime_error{"turn PIMC: root prediction mismatch"};
        validate_value(predictions[0].value);
        const double root_value = std::clamp(double(predictions[0].value), 0.0, 100.0);
        TurnSearch tree{particle, caps, c}; tree.decide(budget, counted);
        result.time_overshoot_count += tree.stats().time_overshoot;
        if (!tree.stats().fallback) {
            values.push_back(turn_first_action_values(particle, tree.root(), root_value));
            result.mean_depth += tree.stats().mean_leaf_depth;
        } else {
            ++result.fallback_count;
            result.fallback_reasons.push_back(tree.stats().reason);
            result.time_fallback_count += tree.stats().reason == "seconds";
            // Same capped particle, now searched by the existing oracle per-action PUCT.
            SearchSettings settings; settings.oracle = true;
            Search<> fallback{particle, {particle}, predictions[0], settings, PuctScore{c}};
            fallback.run(counted, 800);
            ParticleActionValues value; value.root_value = root_value;
            for (const auto& edge : fallback.root().edges)
                value.best[edge.key] = edge.visits ? std::clamp(edge.value_sum / edge.visits, 0.0, 100.0) : root_value;
            values.push_back(std::move(value));
            // No complete macro tree exists on fallback: contribute zero macro depth, flag separately.
        }
    }
    result.scores = mean_particle_scores(keys, values);
    result.action = moves[std::max_element(result.scores.begin(), result.scores.end()) - result.scores.begin()];
    result.mean_depth /= result.particles;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
} // namespace stsrl::pv
