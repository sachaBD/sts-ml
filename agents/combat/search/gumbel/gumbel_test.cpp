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
    EXPECT_TRUE(search.pending().empty());
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

namespace {
std::vector<std::size_t> wave_sizes(Gumbel& search) {
    auto answer = noisy();
    std::vector<std::size_t> waves;
    for (auto queries = search.pending(); !queries.empty(); queries = search.pending()) {
        waves.push_back(queries.size());
        std::vector<Evaluation> evaluations;
        for (const auto& q : queries) evaluations.push_back(answer(q));
        search.evaluated(evaluations);
    }
    return waves;
}
}  // namespace

TEST(Gumbel, EveryRoundFillsWavesOfUpToConsideredActions) {
    auto s = settings(1);
    s.simulations = 256;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    const auto waves = wave_sizes(search);
    EXPECT_EQ(waves.front(), 1u) << "the root comes first, alone";
    EXPECT_LE(std::ranges::max(waves), std::size_t(s.considered_actions));
    EXPECT_LE(waves.size(), 1 + 2 * std::size_t(s.simulations / s.considered_actions)) << "late rounds wait as often as early ones";
}

TEST(Gumbel, SimulationsSharingAWaveSpreadOut) {
    auto root = stsrl::scenarios::jaw_worm(7);
    root.player.energy = 0;  // one move, so every simulation of a wave takes it
    auto s = settings();
    s.particles = 1;
    Gumbel search{root, s, 1};
    EXPECT_GT(std::ranges::max(wave_sizes(search)), 1u);
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
    ASSERT_EQ(search.pending().size(), 1u);
    const auto query = search.pending().front();
    auto answer = flat(query);
    answer.value = 0.37f;
    for (std::size_t i = 0; i < query.actions.size(); ++i)
        if (query.actions[i] == brew) answer.logits[i] = 5.f;

    testing::internal::CaptureStderr();
    search.evaluated({&answer, 1});
    const auto errors = testing::internal::GetCapturedStderr();
    ASSERT_TRUE(search.pending().empty());
    const auto result = search.result();
    EXPECT_FLOAT_EQ(result.value, answer.value);
    EXPECT_EQ(result.action, brew);
    EXPECT_NE(errors.find("ERROR gumbel: cycle truncated"), std::string::npos);
    EXPECT_NE(errors.find("path_action_keys=["), std::string::npos);
    EXPECT_NE(errors.find("BattleContext:"), std::string::npos);
}


namespace {
class GumbelInspection : public testing::Test {
protected:
    const sts::BattleContext state = stsrl::scenarios::jaw_worm(7);
    Gumbel search{state, settings(1), 1};
    Result result;
    Gumbel::Graph graph;
    std::vector<std::pair<std::uint64_t, double>> evaluated;

    void SetUp() override {
        auto answer = noisy();
        result = run(search, [&](const Query& query) {
            auto evaluation = answer(query);
            evaluated.emplace_back(stsrl::combat::observation_key(query.state), evaluation.value);
            return evaluation;
        });
        std::ranges::sort(evaluated);
        graph = search.graph();
    }
};
}  // namespace

TEST_F(GumbelInspection, ExposesTheEvaluatedNodes) {
    std::vector<std::pair<std::uint64_t, double>> reported;
    for (const auto& node : graph.nodes)
        if (!node.terminal) reported.emplace_back(node.observation, node.evaluation_value);
    std::ranges::sort(reported);
    EXPECT_EQ(reported, evaluated);
}

TEST_F(GumbelInspection, ExposesRootActionsInDecisionOrder) {
    const auto root = std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
    ASSERT_NE(root, graph.nodes.end());
    std::vector<std::uint64_t> reported, expected;
    for (const auto& edge : root->edges) reported.push_back(edge.action);
    for (const auto action : result.actions) expected.push_back(stsrl::combat::action_key(state, action));
    EXPECT_EQ(reported, expected);
}

TEST_F(GumbelInspection, ExposesTheDecisionPolicy) {
    const auto root = std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
    ASSERT_NE(root, graph.nodes.end());
    std::vector<float> reported;
    for (const auto& edge : root->edges) reported.push_back(float(edge.improved_policy));
    EXPECT_EQ(reported, result.policy);
}

TEST_F(GumbelInspection, InspectionPreservesTheDecision) {
    EXPECT_EQ(search.result().action, result.action);
    EXPECT_EQ(search.result().policy, result.policy);
    EXPECT_EQ(search.result().value, result.value);
}
