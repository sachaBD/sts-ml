#include "agents/combat/player/decision.hpp"
#include "environments/combat/public.hpp"
#include <functional>
#include <stdexcept>
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
    next_requests();
}

void Decision::evaluated(std::span<const infra::Prediction> predictions) {
    if (predictions.size() != requests_.size()) throw std::invalid_argument{"decision: one prediction per request"};
    for (std::size_t i = 0; i < requests_.size(); ++i) answers_.add(requests_[i], predictions[i]);
    answer(predictions);
    next_requests();
}

void Decision::answer(std::span<const infra::Prediction> predictions) {
    std::vector<search::Evaluation> evaluations;
    auto network = predictions.begin();
    for (const auto* known : from_previous_) evaluations.push_back(nn::NetworkEvaluator::answer(known ? *known : *network++));
    requests_.clear();
    from_previous_.clear();
    search_->evaluated(evaluations);
}

void Decision::next_requests() {
    for (auto queries = search_->pending(); !queries.empty(); queries = search_->pending()) {
        for (const auto& query : queries) {
            auto request = network_.request(query);
            const auto* known = previous_.find(request);
            if (known) answers_.add(request, *known); else requests_.push_back(std::move(request));
            from_previous_.push_back(known);
        }
        if (!requests_.empty()) return;
        answer({});
    }
}
}  // namespace stsrl::player
