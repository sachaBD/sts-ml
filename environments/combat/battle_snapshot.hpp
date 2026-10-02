#pragma once
#include "combat/BattleContext.h"
#include <nlohmann/json.hpp>
namespace stsrl {
// Lossless ordinary initial boundary; callback queues must be empty. RNG is replay-only.
nlohmann::json battle_snapshot(const sts::BattleContext& battle);
sts::BattleContext battle_restore(const nlohmann::json& state);
}
