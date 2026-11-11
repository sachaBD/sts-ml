// Gumbel invariants against fake evaluators: no network, so every expectation follows from the search alone.
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>

using namespace stsrl::search;
using stsrl::combat::legal_actions;

namespace {
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

TEST(Gumbel, OnlyAsksAboutLiveStatesWithTheirLegalMoves) {
    Gumbel search{stsrl::scenarios::jaw_worm(7), settings(1), 1};
    auto answer = noisy();
    run(search, [&](const Query& q) {
        EXPECT_EQ(q.state.outcome, sts::Outcome::UNDECIDED);
        EXPECT_TRUE(std::ranges::equal(q.actions, legal_actions(q.state)));
        return answer(q);
    });
}

TEST(Gumbel, NetworkCallsStayWithinTheSimulationBudget) {
    const auto s = settings(1);
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    int queries = 0;
    auto answer = noisy();
    run(search, [&](const Query& q) { ++queries; return answer(q); });
    EXPECT_LE(queries, s.simulations + 1);
    EXPECT_FALSE(search.pending());
}

TEST(Gumbel, NeverAsksAboutTheSameStateTwice) {
    auto s = settings(1);
    s.simulations = 256;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    std::set<std::uint64_t> asked;
    auto answer = noisy();
    run(search, [&](const Query& q) {
        EXPECT_TRUE(asked.insert(stsrl::combat::observation_key(q.state)).second);
        return answer(q);
    });
}

TEST(Gumbel, ResultIsAProbabilityOverTheLegalMoves) {
    const auto root = stsrl::scenarios::jaw_worm(7);
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

TEST(Gumbel, FlatValuesFollowThePrior) {
    const auto root = stsrl::scenarios::jaw_worm(7);
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
    auto root = stsrl::scenarios::jaw_worm(7);
    root.monsters.arr[0].curHp = 1;
    const auto actions = legal_actions(root);
    const auto loser = std::ranges::find_if(actions, [&](Action a) { return a.getActionType() == sts::search::ActionType::END_TURN; });
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

TEST(Gumbel, ZeroSimulationsPlaysTheNetworkAlone) {
    const auto root = stsrl::scenarios::jaw_worm(7);
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

TEST(Gumbel, ASingleLegalMoveGetsProbabilityOne) {
    auto root = stsrl::scenarios::jaw_worm(7);
    root.player.energy = 0;
    ASSERT_EQ(legal_actions(root).size(), 1u);
    Gumbel search{root, settings(), 1};
    const auto r = run(search, flat);
    EXPECT_EQ(r.action, legal_actions(root).front());
    EXPECT_EQ(r.policy, std::vector<float>{1.f});
}

TEST(Gumbel, EntropicSelfLoopUsesTheNetworkValueAndReportsAnError) {
    auto state = stsrl::scenarios::jaw_worm(13);
    state.potionCapacity = state.potionCount = 2;
    state.potions[0] = sts::Potion::ENTROPIC_BREW;
    state.potions[1] = sts::Potion::COLORLESS_POTION;
    const Action brew{sts::search::ActionType::POTION, 0};
    auto s = settings();
    s.particles = 1;
    s.considered_actions = 1;
    s.simulations = 8;
    Gumbel search{state, s, 1};
    const auto query = search.pending().value();
    auto answer = flat(query);
    answer.value = 0.37f;
    for (std::size_t i = 0; i < query.actions.size(); ++i)
        if (query.actions[i] == brew) answer.logits[i] = 5.f;

    testing::internal::CaptureStderr();
    search.evaluated(answer);
    const auto errors = testing::internal::GetCapturedStderr();
    ASSERT_FALSE(search.pending());
    const auto result = search.result();
    EXPECT_FLOAT_EQ(result.value, answer.value);
    EXPECT_EQ(result.action, brew);
    EXPECT_NE(errors.find("ERROR gumbel: cycle truncated"), std::string::npos);
    EXPECT_NE(errors.find("path_action_keys=["), std::string::npos);
    EXPECT_NE(errors.find("BattleContext:"), std::string::npos);
}


TEST(GumbelGraph, ReportsTheValuesAndCountsUsedByTheSearchWithoutChangingTheResult) {
    for (const auto seed : {1u, 2u, 3u}) for (const int simulations : {0, 8, 64}) {
        auto state = stsrl::scenarios::jaw_worm(seed);
        state.monsters.arr[0].curHp = 1;  // exercises terminal nodes as well as evaluated nodes
        auto s = settings(1);
        s.simulations = simulations;
        Gumbel search{state, s, seed};
        EXPECT_THROW(search.graph(), std::logic_error);
        const auto result = run(search, noisy());
        const auto graph = search.graph();
        EXPECT_EQ(search.result().action, result.action);
        EXPECT_EQ(search.result().policy, result.policy);
        EXPECT_EQ(search.result().value, result.value);
        const auto root = std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
        ASSERT_NE(root, graph.nodes.end());
        ASSERT_EQ(root->edges.size(), result.policy.size());
        EXPECT_EQ(graph.root, stsrl::combat::observation_key(state));
        for (std::size_t i = 0; i < result.policy.size(); ++i) {
            EXPECT_FLOAT_EQ(root->edges[i].improved_policy, result.policy[i]);
            EXPECT_EQ(root->edges[i].action, stsrl::combat::action_key(state, result.actions[i]));
        }
        for (const auto& node : graph.nodes) {
            if (node.terminal) {
                EXPECT_TRUE(node.edges.empty());
                EXPECT_TRUE(node.evaluation_value == 0 || node.evaluation_value == 1);
                EXPECT_EQ(node.mean_value, node.evaluation_value);
                continue;
            }
            int traversals = 0;
            for (const auto& edge : node.edges) {
                int count = 0;
                double weighted = 0;
                for (const auto& child : edge.children) {
                    const auto reached = std::ranges::find(graph.nodes, child.observation, &Gumbel::Graph::Node::observation);
                    ASSERT_NE(reached, graph.nodes.end());
                    weighted += child.visits * reached->mean_value;
                    count += child.visits;
                }
                EXPECT_EQ(count, edge.visits);
                ASSERT_EQ(edge.q.has_value(), count > 0);
                if (count) {
                    EXPECT_NEAR(*edge.q, weighted / count, 1e-12);
                }
                traversals += count;
            }
            EXPECT_EQ(node.visits, traversals + 1);
        }
    }
}
