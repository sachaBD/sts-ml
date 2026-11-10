#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/targets.hpp"
#include "environments/combat/public.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include <gtest/gtest.h>

using namespace stsrl::nn;

namespace {
Fight random_fight(std::uint64_t seed) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, 1};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    Fight fight{battle, seed};
    std::mt19937_64 random{seed};
    while (!fight.over()) {
        const auto actions = stsrl::combat::legal_actions(fight.state());
        fight.play(actions[random() % actions.size()]);
    }
    return fight;
}
}  // namespace

// Rounds partition the fight: losses summed over turn starts are the whole fight's losses.
TEST(Targets, RoundsPartitionTheFight) {
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        const auto fight = random_fight(seed);
        const auto& starts = fight.turn_starts();
        const auto& end = fight.state();
        const auto last = vitals(end);
        ASSERT_FALSE(starts.empty());
        float hp = 0, enemy_hp = 0;
        for (const auto& start : starts) {
            const auto t = targets(start, starts, end);
            hp += t.hp_lost * float(start.max_hp);
            enemy_hp += t.enemy_hp_lost * float(start.enemy_max_hp);
            EXPECT_EQ(t.turns_remaining, float(last.turn - start.turn));
        }
        EXPECT_NEAR(hp, float(starts.front().hp - last.hp), 1e-3) << seed;
        EXPECT_NEAR(enemy_hp, float(starts.front().enemy_hp - last.enemy_hp), 1e-3) << seed;
    }
}

TEST(Targets, OutcomeMatchesTheEnd) {
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        const auto fight = random_fight(seed);
        const auto t = targets(fight.turn_starts().front(), fight.turn_starts(), fight.state());
        const bool won = fight.state().outcome == sts::Outcome::PLAYER_VICTORY;
        EXPECT_EQ(t.won, float(won));
        EXPECT_GE(t.hp, 0);
        EXPECT_LE(t.hp, 1);
        EXPECT_GE(t.enemy_hp, 0);
        EXPECT_LE(t.enemy_hp, 1);
        if (won) EXPECT_EQ(t.enemy_hp, 0);
        if (won) EXPECT_GT(t.hp, 0);
    }
}
