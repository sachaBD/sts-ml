#pragma once
// AlphaGo-style leaf value: (1 - mix) * network + mix * the outcome of a guided rollout from the query's state.
// A function of an answer rather than an evaluator, so it applies equally to batched and inline evaluation.
#include "agents/combat/search/search.hpp"
#include "sim/search/BattleScumSearcher2.h"
#include <cstdint>
#include <random>

namespace stsrl::search {
class RolloutBlend {
public:
    RolloutBlend(const sts::BattleContext& any_state, double mix, std::uint64_t seed);
    Evaluation operator()(const Query&, Evaluation);

private:
    double mix_;
    std::mt19937_64 rng_;
    sts::search::BattleScumSearcher2 rollout_;
};
}  // namespace stsrl::search
