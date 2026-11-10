#include "agents/combat/player/decision.hpp"
#include "environments/combat/public.hpp"

namespace stsrl::player {
Fight::Fight(const sts::BattleContext& start, std::uint64_t seed) : state_{start}, seed_{seed} {
    observe(turn_starts_, state_);
    play_forced();
}

bool Fight::over() const {
    return state_.outcome != sts::Outcome::UNDECIDED || state_.turn >= max_turns || steps_ >= max_steps;
}

void Fight::play(search::Action action) {
    action.execute(state_);
    ++steps_;
    observe(turn_starts_, state_);
    play_forced();
}

void Fight::play_forced() {
    while (!over()) {
        const auto actions = combat::legal_actions(state_);
        if (actions.size() != 1) return;
        actions[0].execute(state_);
        ++steps_;
        observe(turn_starts_, state_);
    }
}

std::mt19937_64 Fight::decision_random() const {
    std::seed_seq seeds{std::uint32_t(seed_), std::uint32_t(seed_ >> 32), std::uint32_t(steps_)};
    return std::mt19937_64{seeds};
}

Decision::Decision(std::unique_ptr<search::CombatSearch> search, const nn::NetworkEvaluator& network)
    : search_{std::move(search)}, network_{network} {
    next_request();
}

void Decision::evaluated(const infra::Prediction& prediction) {
    search_->evaluated(nn::NetworkEvaluator::answer(prediction));
    next_request();
}

void Decision::next_request() {
    if (const auto query = search_->pending()) request_ = network_.request(*query);
}
}  // namespace stsrl::player
