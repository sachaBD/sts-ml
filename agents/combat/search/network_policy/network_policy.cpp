#include "agents/combat/search/network_policy/network_policy.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <cmath>

namespace stsrl::search {
NetworkPolicy::NetworkPolicy(const sts::BattleContext& observed)
    : state_{observed}, actions_{combat::legal_actions(observed)} {}

std::optional<Query> NetworkPolicy::pending() const {
    if (evaluation_) return std::nullopt;
    return Query{state_, actions_};
}

void NetworkPolicy::evaluated(const Evaluation& evaluation) { evaluation_ = evaluation; }

Result NetworkPolicy::result() const {
    const auto& logits = evaluation_->logits;
    const auto best = std::max_element(logits.begin(), logits.end()) - logits.begin();
    std::vector<float> policy(logits.size());
    double sum = 0;
    for (std::size_t i = 0; i < logits.size(); ++i) sum += policy[i] = std::exp(logits[i] - logits[best]);
    for (auto& p : policy) p = float(p / sum);
    return {actions_[best], actions_, std::move(policy), evaluation_->value};
}
}  // namespace stsrl::search
