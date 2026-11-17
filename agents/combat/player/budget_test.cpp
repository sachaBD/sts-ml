#include "agents/combat/player/budget.hpp"
#include <gtest/gtest.h>

using namespace stsrl::player;

TEST(PlayoutCap, SearchesFullyAfterChance) {
    const PlayoutCap cap{256, 32, 0};
    std::mt19937_64 random{1};
    for (int i = 0; i < 100; ++i) EXPECT_EQ(cap({.after_chance = true}, random).simulations, 256);
}

TEST(PlayoutCap, OtherwiseSearchesFullyAtItsShare) {
    std::mt19937_64 random{1};
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ((PlayoutCap{256, 32, 1}({.after_chance = false}, random).simulations), 256);
        EXPECT_EQ((PlayoutCap{256, 32, 0}({.after_chance = false}, random).simulations), 32);
    }
}
