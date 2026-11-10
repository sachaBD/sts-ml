#pragma once
#include "agents/combat/nn/tokens.hpp"

namespace stsrl::nn {
// The inventory_v1 encoding: its layout is layout.toml's [inventory_v1].
infra::Tokens encode_inventory(const sts::BattleContext&, std::span<const sts::search::Action>);
}  // namespace stsrl::nn
