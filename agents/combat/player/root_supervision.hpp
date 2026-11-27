#pragma once
#include "agents/combat/player/decision.hpp"
#include <nlohmann/json_fwd.hpp>
#include <optional>

namespace stsrl::player {
// A side search on the pre-move root. It owns its requests and labels, never the played move.
class RootSupervision {
public:
    using Labels = ThinkingTargets;
    static Labels missing();

    RootSupervision();
    ~RootSupervision();
    void observe(const sts::BattleContext&, std::mt19937_64 random, const nn::NetworkEvaluator&,
                 int step, std::size_t sample);
    std::span<const infra::Tokens> advance();
    bool active() const;
    void evaluated(std::span<const infra::Prediction>);
    // Present for a completed probe, even if every round label is masked.
    std::optional<Labels> labels() const;
    nlohmann::json summary() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::player
