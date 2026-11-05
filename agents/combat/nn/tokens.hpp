#pragma once
// A combat decision as one sequence of tokens (the network's only input).
// Token = [type, id, id2, numeric x 65]; types: 1 context, 2 card, 3 monster, 4 potion, 5 relic, 6 legal action.
// The legal actions are the last tokens, in the order given.
// Action column 67 references the target monster row (0 = no target).
#include "infra/inference/backend.hpp"
#include <span>
#include <string_view>
#include <vector>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"

namespace stsrl::nn {
inline constexpr int width = 68;
enum class Encoding { legacy, inventory_v1 };
Encoding encoding_from_name(std::string_view);

stsrl::infra::Tokens encode(const sts::BattleContext&, std::span<const sts::search::Action> actions,
                            Encoding encoding = Encoding::legacy);
}  // namespace stsrl::nn
