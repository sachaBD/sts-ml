#pragma once
#include "agents/combat/pv/evaluator.hpp"
#include "sim/search/Action.h"
#include <functional>
#include <string>
namespace stsrl::pv {
struct RealTurnCaps {
    std::size_t max_sequences=20000, max_leaves=2048, max_actions=512, max_bytes=256ULL<<20;
    double max_seconds=2;
};
struct RealTurnStats {
    double seconds=0, mean_leaf_depth=0;
    std::size_t sequences=0, leaves=0, network_calls=0, evaluated_states=0;
    std::size_t end_turn_leaves=0, reveal_leaves=0, terminal_leaves=0;
    bool fallback=false, time_overshoot=false;
    std::string reason;
};
struct RealTurnResult { sts::search::Action action; double value=0; RealTurnStats stats; };
using RealTurnEvaluate=std::function<std::vector<Prediction>(std::span<const Inputs>)>;
// Only public signatures determine whether a shared continuation is permissible.
std::string real_turn_public_signature(const sts::BattleContext&);
// All-or-nothing lockstep enumeration; caller supplies ordinary real2000 on any cap.
RealTurnResult decide_real_turn(const sts::BattleContext& observed,
    std::span<const sts::BattleContext> particles, const RealTurnEvaluate&, RealTurnCaps={});
} // namespace stsrl::pv
