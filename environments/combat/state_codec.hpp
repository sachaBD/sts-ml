#pragma once
// A fight state as a couple of hundred bytes that decode on their own. Queued actions are code (std::function),
// not data, so a state holding any cannot be packed.
#include "combat/BattleContext.h"
#include <span>
#include <string>

namespace stsrl::combat {
bool packable(const sts::BattleContext&);
// Appends the state's bytes to `out`. Throws if the state is not packable.
void pack(const sts::BattleContext&, std::string& out);
// Throws unless `bytes` are exactly one packed state.
sts::BattleContext unpack(std::span<const char> bytes);
}  // namespace stsrl::combat
