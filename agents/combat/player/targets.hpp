#pragma once
// Training targets for one decision, derived once its fight is over.
#include "combat/BattleContext.h"
#include <span>
#include <vector>

namespace stsrl::nn {
// Health at one point of a fight; enemy totals cover every monster slot.
struct Vitals {
    int turn, hp, max_hp, enemy_hp, enemy_max_hp;
};
Vitals vitals(const sts::BattleContext&);

// Sample-record order. Fractions are of max HP; a round runs from the decision to the start of our next turn.
struct Targets {
    float won, hp, enemy_hp, hp_lost, enemy_hp_lost, turns_remaining;
};

// Records the state's vitals when it is the first input of a turn not yet recorded; call at every input.
void observe(std::vector<Vitals>& turn_starts, const sts::BattleContext&);

// turn_starts holds the vitals at the first input of each turn, in turn order.
Targets targets(const Vitals& decision, std::span<const Vitals> turn_starts, const sts::BattleContext& end);
}  // namespace stsrl::nn
