#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/targets.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>

using namespace stsrl::player;
using namespace stsrl::nn;
namespace target = stsrl::nn::layout::sample::targets;

namespace {
Fight random_fight(std::uint64_t seed) {
    Fight fight{stsrl::scenarios::jaw_worm(seed), seed};
    std::mt19937_64 random{seed};
    while (!fight.over()) {
        const auto actions = stsrl::combat::legal_actions(fight.state());
        fight.play(actions[random() % actions.size()]);
    }
    return fight;
}
}  // namespace

// Rounds partition the fight: losses summed over turn starts are the whole fight's losses.
TEST(Targets, EveryDecisionBelongsToExactlyOneRound) {
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        const auto fight = random_fight(seed);
        const auto& starts = fight.turn_starts();
        const auto& end = fight.state();
        const auto last = vitals(end);
        ASSERT_FALSE(starts.empty());
        float hp = 0, enemy_hp = 0;
        for (const auto& start : starts) {
            const auto t = targets(start, starts, end);
            hp += t[target::hp_lost] * float(start.max_hp);
            enemy_hp += t[target::enemy_hp_lost] * float(start.enemy_max_hp);
            EXPECT_EQ(t[target::turns_remaining], float(last.turn - start.turn));
        }
        EXPECT_NEAR(hp, float(starts.front().hp - last.hp), 1e-3) << seed;
        EXPECT_NEAR(enemy_hp, float(starts.front().enemy_hp - last.enemy_hp), 1e-3) << seed;
    }
}

TEST(Targets, OutcomeMatchesHowTheFightEnded) {
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        const auto fight = random_fight(seed);
        const auto t = targets(fight.turn_starts().front(), fight.turn_starts(), fight.state());
        const bool won = fight.state().outcome == sts::Outcome::PLAYER_VICTORY;
        EXPECT_EQ(t[target::won], float(won));
        EXPECT_GE(t[target::hp], 0);
        EXPECT_LE(t[target::hp], 1);
        EXPECT_GE(t[target::enemy_hp], 0);
        EXPECT_LE(t[target::enemy_hp], 1);
        if (won) { EXPECT_EQ(t[target::enemy_hp], 0); }
        if (won) { EXPECT_GT(t[target::hp], 0); }
    }
}
