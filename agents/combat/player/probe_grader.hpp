#pragma once
#include "agents/combat/search/search.hpp"
#include "combat/BattleContext.h"
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace stsrl::player {
struct GradeSettings {
    int particles = 8;
    int horizon_turns = 1;  // evaluate at fight end or once state.turn >= root.turn + horizon_turns
    int max_steps = 64;     // caps infinite 0-cost loops
};

// Grades distinct non-prior commits against the prior move on fresh shared particles
// via a greedy-prior plunge, deduplicating network evaluations by observation_key.
class ProbeGrader {
public:
    ProbeGrader(const sts::BattleContext& observed, search::Action prior,
                std::span<const search::Action> candidates, std::uint64_t particle_stream,
                GradeSettings = {});

    std::span<const search::Query> pending() const { return queries_; }
    void evaluated(std::span<const search::Evaluation>);
    // Exact 0.0f for prior; mean_p(score_p(c) - score_p(prior)) for graded candidates.
    std::optional<float> gain(search::Action) const;

private:
    struct Branch { sts::BattleContext state; int steps = 0; bool done = false; };
    struct Pending { std::uint64_t obs; sts::BattleContext state; std::vector<search::Action> actions; };
    struct Cached { double value; std::uint64_t action_key; };

    void advance();
    std::optional<std::size_t> index(search::Action) const;

    GradeSettings settings_;
    int target_turn_;
    std::vector<search::Action> moves_;  // moves_[0] is prior
    std::vector<Branch> branches_;       // [move * particles + particle]
    std::vector<double> scores_;
    std::unordered_map<std::uint64_t, Cached> cache_;
    std::vector<Pending> pending_;
    std::vector<search::Query> queries_;
};
}  // namespace stsrl::player
