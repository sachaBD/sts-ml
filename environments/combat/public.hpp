#pragma once
// Combat as a player sees it: hidden information (draw order, future rolls) is only ever sampled, and moves
// and states are identified by what is public, so a search can share one tree across sampled worlds.
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include <cstdint>
#include <vector>

namespace stsrl::combat {
using sts::search::Action;

// One action per public key: identical card copies in different hand slots are one move.
std::vector<Action> legal_actions(const sts::BattleContext&);
std::uint64_t action_key(const sts::BattleContext&, Action);
std::uint64_t observation_key(const sts::BattleContext&);

// Worlds consistent with what `observed` shows, drawn deterministically from `stream`.
std::vector<sts::BattleContext> particles(const sts::BattleContext& observed, int count, std::uint64_t stream);
std::vector<sts::BattleContext> particles(const sts::BattleContext& observed, int count);  // stream: the observation
}  // namespace stsrl::combat
