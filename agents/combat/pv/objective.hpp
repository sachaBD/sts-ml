#pragma once

#include "combat/BattleContext.h"
#include <cmath>
#include <stdexcept>

namespace stsrl::pv {

// One value unit for recorded targets, network predictions, backups and selection: 100 × win probability.
struct CombatObjective {

    static double terminal_value(const sts::BattleContext& state) {
        if (state.outcome == sts::Outcome::UNDECIDED) {
            throw std::logic_error{"PV objective requires a terminal state"};
        }

        if (state.escapedCombat) {
            throw std::runtime_error{"PV does not support combat escape"};
        }

        if (state.outcome != sts::Outcome::PLAYER_VICTORY) return 0;
        if (state.player.curHp <= 0 || state.potionCount < 0) {
            throw std::runtime_error{"PV: invalid terminal resources"};
        }

        return 100;
    }
};

inline void validate_value(double value) {
    if (!std::isfinite(value) || value < 0) {
        throw std::invalid_argument{"PV value must be finite, nonnegative 100 × win probability"};
    }
}

}  // namespace stsrl::pv
