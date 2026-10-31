#pragma once
// A combat decision as one sequence of tokens (the network's only input).
// Token = [type, id, id2, numeric x 65]; types: 1 context, 2 card, 3 monster, 4 potion, 5 relic, 6 legal action.
// The legal actions are the last tokens, in the order given.
#include "infra/inference/backend.hpp"
#include <span>
#include <vector>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"

namespace stsrl::nn {
inline constexpr int width = 68;

stsrl::infra::Tokens encode(const sts::BattleContext&, std::span<const sts::search::Action> actions);
}  // namespace stsrl::nn
