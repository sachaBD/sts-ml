// Gumbel invariants against fake evaluators: no network, so every expectation follows from the search alone.
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
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

TEST(Gumbel, ZeroSimulationsIsTheNetwork) {
    const auto root = jaw_worm();
    auto s = settings();
    s.simulations = 0;
    Gumbel search{root, s, 1};
    auto answer = noisy();
    std::optional<Evaluation> network;
    const auto r = run(search, [&](const Query& q) { return *(network = answer(q)); });

    const auto& logits = network->logits;
    const auto best = std::ranges::max_element(logits) - logits.begin();
    EXPECT_EQ(r.action, r.actions[best]);
    double sum = 0;
    for (float l : logits) sum += std::exp(double(l) - logits[best]);
    for (std::size_t i = 0; i < logits.size(); ++i)
        EXPECT_NEAR(r.policy[i], std::exp(double(logits[i]) - logits[best]) / sum, 1e-6);
    EXPECT_EQ(r.value, network->value);
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

TEST(GumbelTree, NodeIIsTheIthQuery) {
    for (const std::uint64_t seed : {1, 2, 3}) {
        Gumbel search{jaw_worm(), settings(1), seed};
        std::vector<std::uint64_t> queried;
        auto answer = noisy();
        run(search, [&](const Query& q) { queried.push_back(stsrl::combat::observation_key(q.state)); return answer(q); });
        const auto tree = search.tree();
        ASSERT_EQ(tree.nodes.size(), queried.size());
        std::vector<int> parents(tree.nodes.size(), 0);
        for (const auto& node : tree.nodes)
            for (const auto& edge : node.edges)
                for (const auto& child : edge.children) {
                    EXPECT_EQ(child.observation, queried[child.node]);
                    ++parents[child.node];
                }
        EXPECT_EQ(parents[0], 0);
        for (std::size_t i = 1; i < parents.size(); ++i) EXPECT_EQ(parents[i], 1) << "node " << i;
    }
}

TEST(GumbelTree, AgreesWithTheSearch) {
    for (const std::uint64_t seed : {1, 2, 3}) {
        const auto s = settings(1);
        Gumbel search{jaw_worm(), s, seed};
        const auto r = run(search, noisy());
        const auto tree = search.tree();
        EXPECT_LE(tree.nodes[0].total, s.simulations);
        ASSERT_EQ(tree.nodes[0].edges.size(), r.policy.size());
        for (std::size_t i = 0; i < r.policy.size(); ++i) EXPECT_FLOAT_EQ(tree.nodes[0].edges[i].improved, r.policy[i]);
        for (const auto& node : tree.nodes) {
            int visits = 0;
            for (const auto& edge : node.edges) visits += edge.visits;
            EXPECT_EQ(visits, node.total);
        }
    }
}

TEST(Gumbel, HalvingRoundsAllocateEqually) {
    const auto root = jaw_worm();
    const int A = int(legal_actions(root).size());
    ASSERT_GE(A, 5);
    for (const int sims : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 64, 100, 256}) {
        for (const int considered : {2, 3, 5, 16}) {
            auto s = settings(1);
            s.simulations = sims;
            s.considered_actions = considered;
            Gumbel search{root, s, 1};
            run(search, noisy());
            const auto tree = search.tree();
            const auto& edges = tree.nodes[0].edges;

            const int k = std::min({considered, A, std::max(1, sims / 2)});
            std::vector<int> v;
            for (const auto& e : edges) v.push_back(e.visits);
            std::ranges::sort(v, std::greater{});
            EXPECT_EQ(std::ranges::count_if(v, [](int n) { return n > 0; }), k);

            for (int m = k; m >= 2; m = std::max(2, m / 2)) {
                const int survivors = m == 2 ? 0 : std::max(2, m / 2);
                for (int i = survivors; i < m; ++i)
                    EXPECT_EQ(v[i], v[m - 1]) << "sims=" << sims << " considered=" << considered << " m=" << m;
                if (m == 2) break;
            }
        }
    }
}
