#pragma once
// No search: the network's own choice. The baseline every search is measured against.
#include "agents/combat/search/search.hpp"

namespace stsrl::search {
class NetworkPolicy final : public CombatSearch {
public:
    explicit NetworkPolicy(const sts::BattleContext& observed);
    std::optional<Query> pending() const override;
    void evaluated(const Evaluation&) override;
    Result result() const override;

private:
    sts::BattleContext state_;
    std::vector<Action> actions_;
    std::optional<Evaluation> evaluation_;
};
}  // namespace stsrl::search
