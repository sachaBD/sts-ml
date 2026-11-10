#pragma once
// The entity encoding: its layout is layout.toml's [entity].
#include "agents/combat/nn/tokens.hpp"

namespace stsrl::nn {
infra::Tokens encode_entity(const sts::BattleContext&, std::span<const sts::search::Action>);
}  // namespace stsrl::nn
