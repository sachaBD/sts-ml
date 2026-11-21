#pragma once
#include "combat/BattleContext.h"
#include <span>
#include <string>

namespace stsrl::combat {
// Only bytes produced by this simulator build are supported.
// Appends the state's bytes to `out`.
void pack(const sts::BattleContext&, std::string& out);
// Throws on truncated or trailing bytes.
sts::BattleContext unpack(std::span<const char> bytes);
}  // namespace stsrl::combat
