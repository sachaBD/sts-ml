#include "agents/combat/nn/decision.hpp"
#include "agents/combat/nn/tokens.hpp"
#include "environments/combat/public.hpp"

namespace stsrl::nn {
Fight::Fight(const sts::BattleContext& start, std::uint64_t seed) : state_{start}, seed_{seed} { play_forced(); }

bool Fight::over() const {
    return state_.outcome != sts::Outcome::UNDECIDED || state_.turn >= max_turns || steps_ >= max_steps;
}

void Fight::play(search::Action action) {
    action.execute(state_);
    ++steps_;
    play_forced();
}

void Fight::play_forced() {
    while (!over()) {
        const auto actions = combat::legal_actions(state_);
        if (actions.size() != 1) return;
        actions[0].execute(state_);
        ++steps_;
    }
}

std::mt19937_64 Fight::decision_random() const {
    std::seed_seq seeds{std::uint32_t(seed_), std::uint32_t(seed_ >> 32), std::uint32_t(steps_)};
    return std::mt19937_64{seeds};
}

Decision::Decision(const sts::BattleContext& state, std::unique_ptr<search::CombatSearch> search,
                   double rollout_mix, std::uint64_t rollout_seed)
    : search_{std::move(search)} {
    if (rollout_mix > 0) blend_.emplace(state, rollout_mix, rollout_seed);
    encode_query();
}

void Decision::evaluated(const infra::Prediction& prediction) {
    search::Evaluation evaluation{prediction.value, prediction.logits};
    if (blend_) evaluation = (*blend_)(*search_->pending(), std::move(evaluation));
    search_->evaluated(evaluation);
    encode_query();
}

void Decision::encode_query() {
    if (const auto query = search_->pending()) query_ = encode(query->state, query->actions);
}
}  // namespace stsrl::nn
