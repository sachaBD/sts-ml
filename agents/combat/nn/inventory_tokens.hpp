#pragma once
#include "agents/combat/nn/tokens.hpp"

namespace stsrl::nn {
// inventory_v1 keeps the replay record width, but has its own feature layout and context marker.
// Columns 65..67 are one-based owner/source/target row references; zero means absent.
enum class InventoryKind {
    context = 1, card, monster, potion, relic, action, player, player_effect, monster_effect, inventory
};
inline constexpr int inventory_owner = 65, inventory_source = 66, inventory_target = 67;
inline constexpr int inventory_version = 1;

infra::Tokens encode_inventory(const sts::BattleContext&, std::span<const sts::search::Action>);
}  // namespace stsrl::nn
