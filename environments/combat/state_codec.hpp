#pragma once
#include "combat/BattleContext.h"
#include <span>
#include <string>

namespace stsrl::combat {
// Only bytes produced by this simulator build are supported. Pending action closures cannot be saved.
bool packable(const sts::BattleContext&);
// Appends the state's bytes to `out`. Throws if the state is not packable.
void pack(const sts::BattleContext&, std::string& out);
// Throws on truncated or trailing bytes.
sts::BattleContext unpack(std::span<const char> bytes);
}  // namespace stsrl::combat
