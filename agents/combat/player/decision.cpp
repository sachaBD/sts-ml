#include "agents/combat/player/decision.hpp"
#include "environments/combat/public.hpp"
#include <functional>
#include <string_view>

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

namespace {
// The data's length and the action count fix the shape, so equal keys mean equal requests (up to hash collision).
std::uint64_t key(const infra::Tokens& tokens) {
    const std::string_view bytes{reinterpret_cast<const char*>(tokens.data.data()), tokens.data.size() * sizeof(float)};
    return std::hash<std::string_view>{}(bytes) ^ std::uint64_t(tokens.actions);
}
}  // namespace

const infra::Prediction* Answers::find(const infra::Tokens& request) const {
    const auto found = by_request_.find(key(request));
    return found == by_request_.end() ? nullptr : &found->second;
}

void Answers::add(const infra::Tokens& request, const infra::Prediction& answer) { by_request_.emplace(key(request), answer); }

Decision::Decision(std::unique_ptr<search::CombatSearch> search, const nn::NetworkEvaluator& network, Answers previous)
    : search_{std::move(search)}, network_{network}, previous_{std::move(previous)} {
    next_request();
}

void Decision::evaluated(const infra::Prediction& prediction) {
    answers_.add(request_, prediction);
    search_->evaluated(nn::NetworkEvaluator::answer(prediction));
    next_request();
}

void Decision::next_request() {
    while (const auto query = search_->pending()) {
        request_ = network_.request(*query);
        const auto* known = previous_.find(request_);
        if (!known) return;
        answers_.add(request_, *known);
        search_->evaluated(nn::NetworkEvaluator::answer(*known));
    }
}
}  // namespace stsrl::player
