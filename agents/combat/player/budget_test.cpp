#include "agents/combat/player/budget.hpp"
#include <gtest/gtest.h>

using namespace stsrl::player;

TEST(PlayoutCap, SearchesFullyAfterChance) {
    const PlayoutCap cap{256, 32, 0};
    std::mt19937_64 random{1};
    for (int i = 0; i < 100; ++i) {
        const auto budget = cap({.after_chance = true}, random);
        EXPECT_EQ(budget.simulations, 256);
        EXPECT_TRUE(budget.full);
    }
}

TEST(PlayoutCap, OtherwiseSearchesFullyAtItsShare) {
    std::mt19937_64 random{1};
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ((PlayoutCap{256, 32, 1}({.after_chance = false}, random).simulations), 256);
        EXPECT_EQ((PlayoutCap{256, 32, 0}({.after_chance = false}, random).simulations), 32);
    }
}

TEST(PlayoutCap, FullSearchesAreAboutTheirShareAwayFromChance) {
    const PlayoutCap cap{256, 32, 0.2};
    std::mt19937_64 random{1};
    int full = 0;
    for (int i = 0; i < 10000; ++i) full += cap({.after_chance = false}, random).full;
    EXPECT_NEAR(full / 10000., 0.2, 0.02);
}
