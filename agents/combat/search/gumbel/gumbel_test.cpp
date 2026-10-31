// Gumbel invariants against fake evaluators: no network, so every expectation follows from the search alone.
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <random>

using namespace stsrl::search;
using stsrl::combat::legal_actions;

namespace {
sts::BattleContext jaw_worm() {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, 7, 1};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    return battle;
}

bool wins_now(const sts::BattleContext& state, Action action) {
    auto next = state;
    action.execute(next);
    return next.outcome == sts::Outcome::PLAYER_VICTORY;
}

Evaluation flat(const Query& q) { return {0.5f, std::vector<float>(q.actions.size(), 0.f)}; }

// Arbitrary but repeatable answers, so results are not trivially uniform.
auto noisy() {
    return [rng = std::mt19937{1}](const Query& q) mutable {
        std::normal_distribution<float> logit;
        Evaluation e{std::uniform_real_distribution<float>{}(rng), {}};
        for (std::size_t i = 0; i < q.actions.size(); ++i) e.logits.push_back(logit(rng));
        return e;
    };
}

Gumbel::Settings settings(double gumbel_scale = 0) {
    Gumbel::Settings s;
    s.simulations = 64;
    s.particles = 4;
    s.gumbel_scale = gumbel_scale;
    return s;
}
}  // namespace

TEST(Gumbel, QueriesAreLiveLegalDecisions) {
    Gumbel search{jaw_worm(), settings(1), 1};
    auto answer = noisy();
    run(search, [&](const Query& q) {
        EXPECT_EQ(q.state.outcome, sts::Outcome::UNDECIDED);
        EXPECT_TRUE(std::ranges::equal(q.actions, legal_actions(q.state)));
        return answer(q);
    });
}

TEST(Gumbel, AsksAtMostOncePerSimulationPlusRoot) {
    const auto s = settings(1);
    Gumbel search{jaw_worm(), s, 1};
    int queries = 0;
    auto answer = noisy();
    run(search, [&](const Query& q) { ++queries; return answer(q); });
    EXPECT_LE(queries, s.simulations + 1);
    EXPECT_FALSE(search.pending());
}

TEST(Gumbel, ResultIsAPolicyOverTheRootActions) {
    const auto root = jaw_worm();
    Gumbel search{root, settings(1), 1};
    const auto r = run(search, noisy());
    EXPECT_TRUE(std::ranges::equal(r.actions, legal_actions(root)));
    ASSERT_EQ(r.policy.size(), r.actions.size());
    EXPECT_TRUE(std::ranges::all_of(r.policy, [](float p) { return p >= 0; }));
    EXPECT_NEAR(std::accumulate(r.policy.begin(), r.policy.end(), 0.0), 1.0, 1e-5);
    EXPECT_NE(std::ranges::find(r.actions, r.action), r.actions.end());
    EXPECT_GE(r.value, 0);
    EXPECT_LE(r.value, 1);
}

TEST(Gumbel, SameSeedSameResult) {
    Gumbel a{jaw_worm(), settings(1), 3}, b{jaw_worm(), settings(1), 3};
    const auto ra = run(a, noisy()), rb = run(b, noisy());
    EXPECT_EQ(ra.action, rb.action);
    EXPECT_EQ(ra.policy, rb.policy);
    EXPECT_EQ(ra.value, rb.value);
}

TEST(Gumbel, FlatValuesFollowThePrior) {
    const auto root = jaw_worm();
    const auto favourite = legal_actions(root).back();
    Gumbel search{root, settings(), 1};
    const auto r = run(search, [&](const Query& q) {
        auto e = flat(q);
        for (std::size_t i = 0; i < q.actions.size(); ++i) e.logits[i] = q.actions[i] == favourite ? 5.f : 0.f;
        return e;
    });
    EXPECT_EQ(r.action, favourite);
}

TEST(Gumbel, WinningMoveBeatsAWrongPrior) {
    auto root = jaw_worm();
    root.monsters.arr[0].curHp = 1;
    const auto actions = legal_actions(root);
    const auto loser = std::ranges::find_if(actions, [&](Action a) { return !wins_now(root, a); });
    ASSERT_TRUE(std::ranges::any_of(actions, [&](Action a) { return wins_now(root, a); }));
    ASSERT_NE(loser, actions.end());

    Gumbel search{root, settings(), 1};
    const auto r = run(search, [&](const Query& q) {
        auto e = flat(q);
        for (std::size_t i = 0; i < q.actions.size(); ++i) e.logits[i] = q.actions[i] == *loser ? 3.f : 0.f;
        return e;
    });
    EXPECT_TRUE(wins_now(root, r.action));
}

TEST(Gumbel, ForcedMoveIsCertain) {
    auto root = jaw_worm();
    root.player.energy = 0;
    ASSERT_EQ(legal_actions(root).size(), 1u);
    Gumbel search{root, settings(), 1};
    const auto r = run(search, flat);
    EXPECT_EQ(r.action, legal_actions(root).front());
    EXPECT_EQ(r.policy, std::vector<float>{1.f});
}
