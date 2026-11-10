#pragma once
// A combat decision as one sequence of width-68 tokens (the network's only input); the legal actions are the
// last tokens, in the order given. Each model owns its layout: entity/encoding.hpp, inventory/encoding.hpp.
#include "infra/inference/backend.hpp"
#include <span>
#include <string_view>
#include <vector>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"

namespace stsrl::nn {
inline constexpr int width = 68;
enum class Encoding { entity, inventory_v1 };
Encoding encoding_from_name(std::string_view);

stsrl::infra::Tokens encode(const sts::BattleContext&, std::span<const sts::search::Action> actions,
                            Encoding encoding);
}  // namespace stsrl::nn
