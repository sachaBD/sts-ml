#pragma once
#include "agents/combat/pv/turn_search.hpp"
namespace stsrl::pv {
struct TurnTargetChild { std::uint32_t action; std::size_t visits; double value; };
struct TurnPrefixTarget { double value = 0; std::size_t visits = 0; std::vector<TurnTargetChild> children; };
// Exact played prefix; next moves grouped/mapped to the canonical PV legal menu.
TurnPrefixTarget turn_prefix_target(const TurnNode&, std::span<const std::uint32_t> prefix,
                                   const sts::BattleContext& current);
} // namespace stsrl::pv
