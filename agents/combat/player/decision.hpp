#pragma once
#include "agents/combat/nn/evaluator.hpp"
#include "agents/combat/player/targets.hpp"
#include "agents/combat/search/search.hpp"
#include <cstdint>
#include <memory>
#include <random>

namespace stsrl::player {
class Fight {
public:
    static constexpr int max_turns = 50, max_steps = 512;  // an undecided fight past either is lost

    Fight(const sts::BattleContext& start, std::uint64_t seed);
    const sts::BattleContext& state() const { return state_; }
    int steps() const { return steps_; }  // forced moves included
    const std::vector<Vitals>& turn_starts() const { return turn_starts_; }
    bool over() const;
    void play(search::Action);  // then every forced move
    // Depends on the fight and step alone, so a decision replays from its state without the ones before it.
    std::mt19937_64 decision_random() const;

private:
    void play_forced();
    sts::BattleContext state_;
    std::uint64_t seed_;
    int steps_ = 0;
    std::vector<Vitals> turn_starts_;
};

// One decision: a search whose queries the network answers. pending() is the network request, if any.
class Decision {
public:
    Decision(std::unique_ptr<search::CombatSearch>, const nn::NetworkEvaluator&);
    const infra::Tokens* pending() const { return search_->pending() ? &request_ : nullptr; }
    void evaluated(const infra::Prediction& prediction);
    search::Result result() const { return search_->result(); }

private:
    void next_request();
    std::unique_ptr<search::CombatSearch> search_;
    const nn::NetworkEvaluator& network_;
    infra::Tokens request_;
};
}  // namespace stsrl::player
