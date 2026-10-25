#pragma once
// A combat decision as one sequence of tokens (the network's only input).
// Token = [type, id, id2, numeric x 65]; types: 1 context, 2 card, 3 monster, 4 potion, 5 relic, 6 legal action.
// The legal actions are the last tokens, in legal_actions() order.
#include <span>
#include <vector>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"

namespace stsrl::nn {
inline constexpr int width = 68;

struct Tokens {
    std::vector<float> data;  // count x width
    int count = 0, actions = 0;
};

// Every legal action, one per public action key (duplicate card copies merged).
std::vector<sts::search::Action> legal_actions(const sts::BattleContext&);
Tokens encode(const sts::BattleContext&, std::span<const sts::search::Action> actions);
}  // namespace stsrl::nn
