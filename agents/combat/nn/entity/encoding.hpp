#pragma once
// The entity encoding: its layout is entity/layout.toml.
#include "agents/combat/nn/tokens.hpp"

namespace stsrl::nn {
std::unique_ptr<const Encoder> entity_encoder(Layout);
}  // namespace stsrl::nn
