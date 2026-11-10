#pragma once
// The entity model's tokens: 1 context, 2 card, 3 monster, 4 potion, 5 relic, 6 legal action.
// Token = [type, id, id2, numeric x 65]. Action column 67 references the target monster row (0 = no target).
#include "agents/combat/nn/tokens.hpp"

namespace stsrl::nn {
infra::Tokens encode_entity(const sts::BattleContext&, std::span<const sts::search::Action>);
}  // namespace stsrl::nn
