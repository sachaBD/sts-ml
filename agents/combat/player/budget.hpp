#pragma once
// How much search each self-play decision gets.
#include <random>

namespace stsrl::player {
struct Budget {
    int simulations;
};

// What the driver knows before it searches a decision.
struct Moment {
    bool after_chance;  // a chance transition since the last decision: End Turn, a draw, a random effect
};

// Full search after chance; otherwise full with probability full_share, else cheap.
struct PlayoutCap {
    int full, cheap;
    double full_share;

    void validate() const;  // throws on settings out of range
    Budget operator()(const Moment&, std::mt19937_64& random) const;
};

using BudgetRule = PlayoutCap;  // the budget rule self-play uses
}  // namespace stsrl::player
