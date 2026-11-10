#pragma once
// The inventory_v1 encoding: its layout is inventory/layout.toml.
#include "agents/combat/nn/tokens.hpp"

namespace stsrl::nn {
std::unique_ptr<const Encoder> inventory_encoder(Layout);
}  // namespace stsrl::nn
