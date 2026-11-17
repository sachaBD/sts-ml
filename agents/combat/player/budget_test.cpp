#include "agents/combat/player/budget.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace stsrl::player;

TEST(PlayoutCap, SearchesFullyAfterChance) {
    const PlayoutCap cap{256, 32, 0};
    std::mt19937_64 random{1};
    EXPECT_EQ(cap({.after_chance = true}, random).simulations, cap.full);
}

TEST(PlayoutCap, OtherwiseSearchesFullyAtItsShare) {
    std::mt19937_64 random{1};
    const PlayoutCap always_full{256, 32, 1}, always_cheap{256, 32, 0};
    EXPECT_EQ(always_full({.after_chance = false}, random).simulations, always_full.full);
    EXPECT_EQ(always_cheap({.after_chance = false}, random).simulations, always_cheap.cheap);
}

TEST(PlayoutCap, MixedBudgetsReplayFromTheSameRandomState) {
    const PlayoutCap cap{256, 32, 0.5};
    std::mt19937_64 random{1}, replay{1};
    bool saw_full = false, saw_cheap = false;
    for (int i = 0; i < 64; ++i) {
        const auto simulations = cap({.after_chance = false}, random).simulations;
        EXPECT_EQ(simulations, cap({.after_chance = false}, replay).simulations);
        EXPECT_TRUE(simulations == cap.full || simulations == cap.cheap);
        saw_full |= simulations == cap.full;
        saw_cheap |= simulations == cap.cheap;
    }
    EXPECT_TRUE(saw_full);
    EXPECT_TRUE(saw_cheap);
}

TEST(PlayoutCap, RejectsInvalidSettings) {
    for (const auto cap : {PlayoutCap{-1, 32, 0.5}, PlayoutCap{256, -1, 0.5},
                           PlayoutCap{256, 32, -0.1}, PlayoutCap{256, 32, 1.1},
                           PlayoutCap{256, 32, std::numeric_limits<double>::quiet_NaN()}})
        EXPECT_THROW(cap.validate(), std::invalid_argument);
    EXPECT_NO_THROW((PlayoutCap{0, 0, 0}.validate()));
    EXPECT_NO_THROW((PlayoutCap{256, 32, 1}.validate()));
}
