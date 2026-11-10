#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/nn/entity/encoding.hpp"
#include "agents/combat/nn/inventory/encoding.hpp"
#include <stdexcept>
#include <string>

namespace stsrl::nn {
Encoding encoding_from_name(std::string_view name) {
    if (name == "legacy") return Encoding::legacy;
    if (name == "inventory_v1") return Encoding::inventory_v1;
    throw std::invalid_argument{"unknown combat token encoding: " + std::string{name}};
}

stsrl::infra::Tokens encode(const sts::BattleContext& bc, std::span<const sts::search::Action> actions,
                            Encoding encoding) {
    return encoding == Encoding::inventory_v1 ? encode_inventory(bc, actions) : encode_entity(bc, actions);
}
}  // namespace stsrl::nn
