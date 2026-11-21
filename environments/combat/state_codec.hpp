#pragma once
#include "combat/BattleContext.h"
#include <span>
#include <string>

namespace stsrl::combat {
// Native scalar bytes in simulator field order, without padding or compression. Trusted, same-build data only:
// this is not a portable archive format or a validator for arbitrary bytes. Queued closures cannot be packed.
bool packable(const sts::BattleContext&);
// Appends the state's bytes to `out`. Throws if the state is not packable.
void pack(const sts::BattleContext&, std::string& out);
// Throws on truncated or trailing bytes.
sts::BattleContext unpack(std::span<const char> bytes);
}  // namespace stsrl::combat
