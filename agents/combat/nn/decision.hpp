#pragma once
#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include "agents/combat/search/search.hpp"
#include "infra/inference/backend.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <random>

namespace stsrl::nn {
class Fight {
public:
    static constexpr int max_turns = 50, max_steps = 512;  // an undecided fight past either is lost

    Fight(const sts::BattleContext& start, std::uint64_t seed);
    const sts::BattleContext& state() const { return state_; }
    int steps() const { return steps_; }  // forced moves included
    bool over() const;
    void play(search::Action);  // then every forced move
    // Depends on the fight and step alone, so a decision replays from its state without the ones before it.
    std::mt19937_64 decision_random() const;

private:
    void play_forced();
    sts::BattleContext state_;
    std::uint64_t seed_;
    int steps_ = 0;
};

// A search answered by the network, with leaf values blended with guided rollouts when rollout_mix > 0.
class Decision {
public:
    Decision(const sts::BattleContext& state, std::unique_ptr<search::CombatSearch>, double rollout_mix,
             std::uint64_t rollout_seed);
    const infra::Tokens* pending() const { return search_->pending() ? &query_ : nullptr; }
    void evaluated(const infra::Prediction&);
    search::Result result() const { return search_->result(); }

private:
    void encode_query();
    std::unique_ptr<search::CombatSearch> search_;
    std::optional<search::RolloutBlend> blend_;
    infra::Tokens query_;
};
}  // namespace stsrl::nn
