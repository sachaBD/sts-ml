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

bool loses_now(const sts::BattleContext& state, Action action) {
    auto next = state;
    action.execute(next);
    return next.outcome == sts::Outcome::PLAYER_LOSS;
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

// The worlds the search asks about: a sampled world carries its own seed.
std::set<std::uint64_t> worlds(std::uint64_t world_draw) {
    auto s = settings();
    s.world_draw = world_draw;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    std::set<std::uint64_t> seeds;
    auto answer = noisy();
    run(search, [&](const Query& q) {
        seeds.insert(q.state.seed);
        return answer(q);
    });
    return seeds;
}

TEST(Gumbel, TheWorldDrawChoosesTheSampledWorlds) {
    EXPECT_EQ(worlds(0), worlds(0));
    const auto other = worlds(1);
    EXPECT_GT(other.size(), 1u);
    for (const auto seed : worlds(0)) EXPECT_EQ(other.contains(seed), seed == stsrl::scenarios::jaw_worm(7).seed);
}

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

TEST(Gumbel, AWaveJoinsPassesUpToTheWaveSize) {
    auto s = settings(1);
    s.considered_actions = 4;
    s.wave = 8;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    const auto waves = wave_sizes(search);
    EXPECT_EQ(waves.front(), 1u) << "the root comes first, alone";
    EXPECT_GT(std::ranges::max(waves), std::size_t(s.considered_actions));
    EXPECT_LE(std::ranges::max(waves), std::size_t(s.wave));
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
    root.player.curHp = 1;
    const auto actions = legal_actions(root);
    const auto loser = std::ranges::find_if(actions, [&](Action a) { return a.getActionType() == sts::search::ActionType::END_TURN; });
    ASSERT_TRUE(std::ranges::any_of(actions, [&](Action a) { return wins_now(root, a); }));
    ASSERT_NE(loser, actions.end());
    ASSERT_TRUE(loses_now(root, *loser));

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

TEST(Gumbel, UnitNoiseWithZeroSimulationsSamplesThePrior) {
    const auto root = stsrl::scenarios::jaw_worm(7);
    auto s = settings(1);
    s.simulations = 0;
    s.particles = 1;
    const auto actions = legal_actions(root);
    std::vector<int> selected(actions.size());
    std::vector<float> prior;
    constexpr int draws = 4096;
    for (int seed = 0; seed < draws; ++seed) {
        Gumbel search{root, s, std::uint64_t(seed)};
        int queries = 0;
        const auto result = run(search, [&](const Query& q) {
            ++queries;
            auto e = flat(q);
            e.logits.front() = 1;  // nonuniform prior
            return e;
        });
        ASSERT_EQ(queries, 1);
        const auto chosen = std::ranges::find(actions, result.action);
        ASSERT_NE(chosen, actions.end());
        ++selected[chosen - actions.begin()];
        prior = result.policy;
    }
    for (std::size_t i = 0; i < prior.size(); ++i) {
        const double p = prior[i], stddev = std::sqrt(draws * p * (1 - p));
        EXPECT_NEAR(selected[i], draws * p, 6 * stddev);
    }
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

namespace {
const Action brew{sts::search::ActionType::POTION, 0};

// With no free potion slot, drinking Entropic Brew returns to the same observation.
sts::BattleContext brew_loop() {
    auto state = stsrl::scenarios::jaw_worm(13);
    state.seed = 52;  // a fight whose sampled worlds brew Entropic Brew again
    state.potionCapacity = state.potionCount = 2;
    state.potions[0] = sts::Potion::ENTROPIC_BREW;
    state.potions[1] = sts::Potion::COLORLESS_POTION;
    return state;
}
}  // namespace

TEST(Gumbel, EntropicSelfLoopUsesTheNetworkValueAndReportsAnError) {
    const auto state = brew_loop();
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
    const auto graph = search.graph();
    const auto root = std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
    ASSERT_NE(root, graph.nodes.end());
    const auto edge = std::ranges::find(root->edges, stsrl::combat::action_key(state, brew), &Gumbel::Graph::Edge::action);
    ASSERT_NE(edge, root->edges.end());
    EXPECT_EQ(edge->transition, Gumbel::TransitionKind::chance);
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

TEST_F(GumbelInspection, OnlyExecutedEdgesHaveAKnownTransition) {
    for (const auto& node : graph.nodes)
        for (const auto& edge : node.edges)
            EXPECT_EQ(edge.transition == Gumbel::TransitionKind::unknown, edge.visits == 0);
}

// The value rules, checked on every node, action and link of a search that meets a cycle and chance.
namespace {
class GumbelValues : public testing::Test {
protected:
    Result result;
    Gumbel::Graph graph;

    void SetUp() override {
        Gumbel search{brew_loop(), settings(1), 1};
        testing::internal::CaptureStderr();
        result = run(search, noisy());
        testing::internal::GetCapturedStderr();
        graph = search.graph();
    }

    const Gumbel::Graph::Node& node(std::uint64_t observation) const {
        return *std::ranges::find(graph.nodes, observation, &Gumbel::Graph::Node::observation);
    }

    double link_value(const Gumbel::Graph::Child& link) const {
        return link.bootstrap ? node(link.observation).evaluation_value : node(link.observation).backed_value;
    }

    bool reaches(std::uint64_t from, std::uint64_t to) const {
        std::vector<std::uint64_t> pending{from};
        std::set<std::uint64_t> seen;
        while (!pending.empty()) {
            const auto at = pending.back();
            pending.pop_back();
            if (at == to) return true;
            if (!seen.insert(at).second) continue;
            for (const auto& edge : node(at).edges)
                for (const auto& link : edge.children)
                    if (!link.bootstrap) pending.push_back(link.observation);
        }
        return false;
    }
};
}  // namespace

TEST_F(GumbelValues, NodesTakeTheBestOfTheirVisitedActions) {
    for (const auto& n : graph.nodes) {
        double best = -INFINITY;
        for (const auto& edge : n.edges)
            if (edge.q) best = std::max(best, *edge.q);
        EXPECT_EQ(n.backed_value, best == -INFINITY ? n.evaluation_value : best);
    }
}

TEST_F(GumbelValues, ActionsAverageTheirLinks) {
    int mixed = 0;
    for (const auto& n : graph.nodes)
        for (const auto& edge : n.edges) {
            if (!edge.visits) continue;
            double sum = 0;
            for (const auto& link : edge.children) sum += link.visits * link_value(link);
            EXPECT_DOUBLE_EQ(*edge.q, sum / edge.visits);
            mixed += edge.children.size() > 1;
        }
    EXPECT_GT(mixed, 0) << "some action must reach several outcomes";
}

TEST_F(GumbelValues, LinksBootstrapExactlyWhenTheyCloseACycle) {
    int bootstraps = 0;
    for (const auto& n : graph.nodes)
        for (const auto& edge : n.edges)
            for (const auto& link : edge.children) {
                EXPECT_EQ(link.bootstrap, reaches(link.observation, n.observation));
                bootstraps += link.bootstrap;
            }
    EXPECT_GT(bootstraps, 0) << "the search must meet a cycle";
}

TEST_F(GumbelValues, TheDecisionValueIsTheRootValue) {
    EXPECT_FLOAT_EQ(result.value, node(graph.root).backed_value);
}

TEST(Gumbel, ZeroSimulationsLeavesEveryTransitionUnknown) {
    auto s = settings();
    s.simulations = 0;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    run(search, flat);
    const auto graph = search.graph();
    ASSERT_EQ(graph.nodes.size(), 1u);
    ASSERT_FALSE(graph.nodes.front().edges.empty());
    for (const auto& edge : graph.nodes.front().edges)
        EXPECT_EQ(edge.transition, Gumbel::TransitionKind::unknown);
}

TEST(Gumbel, ClassifiesTheExecutionRatherThanTheCard) {
    for (bool blocked : {false, true}) {
        auto state = stsrl::scenarios::jaw_worm(7);
        state.cards = sts::CardManager{};
        state.cards.createTempCardInHand(sts::CardInstance{sts::CardId::STRIKE_RED});
        state.cards.createTempCardInHand(sts::CardInstance{sts::CardId::SHRUG_IT_OFF});
        state.cards.createTempCardInDrawPile(0, sts::CardInstance{sts::CardId::BASH});
        state.cards.createTempCardInDrawPile(1, sts::CardInstance{sts::CardId::DEFEND_RED});
        if (blocked) state.player.debuff<PlayerStatus::NO_DRAW>(1);
        Gumbel search{state, settings(), 1};
        run(search, flat);
        const auto graph = search.graph();
        const auto root = std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
        ASSERT_NE(root, graph.nodes.end());
        for (const auto action : legal_actions(state)) {
            const auto edge = std::ranges::find(root->edges, stsrl::combat::action_key(state, action),
                                                &Gumbel::Graph::Edge::action);
            ASSERT_NE(edge, root->edges.end());
            ASSERT_GT(edge->visits, 0);
            auto next = state;
            const auto effects = action.execute(next);
            EXPECT_EQ(edge->transition, effects.isChance() ? Gumbel::TransitionKind::chance
                                                          : Gumbel::TransitionKind::deterministic);
        }
    }
}

namespace {
class GumbelTerminalTransition : public testing::Test {
protected:
    sts::BattleContext state = stsrl::scenarios::jaw_worm(7);

    void SetUp() override {
        state.cards = sts::CardManager{};
        state.cards.createTempCardInHand(sts::CardInstance{sts::CardId::STRIKE_RED});
        state.monsters.arr[0].curHp = 1;
    }

    Gumbel::Graph search() {
        auto s = settings();
        s.simulations = 2;
        s.particles = 2;
        s.considered_actions = 1;
        Gumbel search{state, s, 1};
        run(search, [](const Query& query) {
            auto evaluation = flat(query);
            for (std::size_t i = 0; i < query.actions.size(); ++i)
                if (query.actions[i].getActionType() == sts::search::ActionType::CARD) evaluation.logits[i] = 5;
            return evaluation;
        });
        return search.graph();
    }
};
}  // namespace

TEST(GumbelTransition, ChanceCannotBeDowngraded) {
    for (bool first : {false, true}) {
        for (bool later : {false, true}) {
            auto kind = Gumbel::classify_transition(Gumbel::TransitionKind::unknown, first);
            kind = Gumbel::classify_transition(kind, later);
            EXPECT_EQ(kind == Gumbel::TransitionKind::chance, first || later);
        }
    }
}

TEST_F(GumbelTerminalTransition, MatchingOutcomesDoNotImplyDeterminism) {
    state.cards = sts::CardManager{};
    state.cards.createTempCardInHand(sts::CardInstance{sts::CardId::SWORD_BOOMERANG});
    const auto graph = search();
    const auto root = std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
    ASSERT_NE(root, graph.nodes.end());
    const auto edge = std::ranges::find_if(root->edges, [](const auto& e) { return e.visits > 0; });
    ASSERT_NE(edge, root->edges.end());
    ASSERT_GT(edge->visits, 1);
    ASSERT_EQ(edge->children.size(), 1u);
    EXPECT_EQ(edge->transition, Gumbel::TransitionKind::chance);
}

TEST_F(GumbelTerminalTransition, TerminalTransitionsAreClassified) {
    const auto graph = search();
    const auto root = std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
    ASSERT_NE(root, graph.nodes.end());
    const auto edge = std::ranges::find_if(root->edges, [](const auto& e) { return e.visits > 0; });
    ASSERT_NE(edge, root->edges.end());
    ASSERT_EQ(edge->children.size(), 1u);
    const auto reached = std::ranges::find(graph.nodes, edge->children.front().observation,
                                          &Gumbel::Graph::Node::observation);
    ASSERT_NE(reached, graph.nodes.end());
    ASSERT_TRUE(reached->terminal);
    EXPECT_EQ(edge->transition, Gumbel::TransitionKind::deterministic);
}

// Steering: the caller picks each simulation's path and world.
namespace {
Gumbel steered(const sts::BattleContext& state) {
    auto s = settings();
    s.simulations = 0;
    return Gumbel{state, s, 1};
}

void answer_pending(Gumbel& search) {
    std::vector<Evaluation> evaluations;
    auto answer = noisy();
    for (auto queries = search.pending(); !queries.empty(); queries = search.pending()) {
        evaluations.clear();
        for (const auto& q : queries) evaluations.push_back(answer(q));
        search.evaluated(evaluations);
    }
}

const Gumbel::Graph::Node& root_node(const Gumbel::Graph& graph) {
    return *std::ranges::find(graph.nodes, graph.root, &Gumbel::Graph::Node::observation);
}

std::set<std::uint64_t> steered_worlds(int world) {
    auto search = steered(stsrl::scenarios::jaw_worm(7));
    answer_pending(search);
    std::set<std::uint64_t> seeds;
    const std::vector<Gumbel::Pick> picks{{{0}, world}};
    search.simulate(picks);
    for (const auto& q : search.pending()) seeds.insert(q.state.seed);
    return seeds;
}
}  // namespace

TEST(GumbelSteering, VisitsOnlyTheChosenMove) {
    auto search = steered(stsrl::scenarios::jaw_worm(7));
    answer_pending(search);
    const std::vector<Gumbel::Pick> picks{{{1}, 5, false}};
    search.simulate(picks);
    answer_pending(search);
    const auto graph = search.graph();
    const auto& root = root_node(graph);
    for (std::size_t i = 0; i < root.edges.size(); ++i) EXPECT_EQ(root.edges[i].visits, i == 1);
    const auto reached = search.state(std::vector<std::size_t>{1}, 5);
    EXPECT_EQ(stsrl::combat::observation_key(reached), root.edges[1].children.front().observation);
}

TEST(GumbelSteering, WorldsContinueTheScheduledStream) {
    const auto scheduled = worlds(0);  // the scheduled search samples worlds 0-3
    for (const int world : {0, 3}) EXPECT_TRUE(std::ranges::includes(scheduled, steered_worlds(world)));
    EXPECT_FALSE(std::ranges::includes(scheduled, steered_worlds(4)));
}

TEST(GumbelSteering, SteersOnTopOfAScheduledSearch) {
    auto s = settings();
    s.simulations = 8;
    s.particles = 2;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    answer_pending(search);
    const int before = root_node(search.graph()).edges[0].visits;
    const std::vector<Gumbel::Pick> picks{{{0}, 5, false}};
    search.simulate(picks);
    answer_pending(search);
    EXPECT_EQ(root_node(search.graph()).edges[0].visits, before + 1);
}

TEST(GumbelSteering, InvalidSteerLeavesSearchUsable) {
    auto search = steered(stsrl::scenarios::jaw_worm(7));
    answer_pending(search);
    const std::vector<Gumbel::Pick> valid{{{0}, 0, false}};
    search.simulate(valid);
    answer_pending(search);
    const std::vector<Gumbel::Pick> bad{{{0}, 0, false}, {{0, 999}, 0, false}};
    EXPECT_THROW(search.simulate(bad), std::invalid_argument);
    EXPECT_TRUE(search.pending().empty());
    EXPECT_EQ(root_node(search.graph()).edges[0].visits, 1);
    search.simulate(valid);
    answer_pending(search);
    EXPECT_EQ(root_node(search.graph()).edges[0].visits, 2);
}

TEST(GumbelWorldTracking, OffByDefaultLeavesLinkWorldsEmpty) {
    Gumbel search{stsrl::scenarios::jaw_worm(7), settings(), 1};
    answer_pending(search);
    for (const auto& node : search.graph().nodes)
        for (const auto& edge : node.edges)
            for (const auto& child : edge.children) EXPECT_TRUE(child.worlds.empty());
}

TEST(GumbelWorldTracking, RecordsPerWorldVisitsSummingToLinkVisits) {
    auto s = settings();
    s.simulations = 16;
    s.particles = 3;
    s.record_worlds = true;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    answer_pending(search);
    const std::vector<Gumbel::Pick> picks{{{0}, 5, false}};
    search.simulate(picks);
    answer_pending(search);
    const auto graph = search.graph();
    for (const auto& node : graph.nodes)
        for (const auto& edge : node.edges)
            for (const auto& child : edge.children) {
                std::set<int> seen;
                int sum = 0;
                for (const auto& w : child.worlds) {
                    EXPECT_TRUE(seen.insert(w.world).second);
                    EXPECT_GT(w.visits, 0);
                    sum += w.visits;
                }
                EXPECT_EQ(sum, child.visits);
            }
    const auto& root = root_node(graph);
    EXPECT_TRUE(std::ranges::any_of(root.edges[0].children, [](const auto& c) {
        return std::ranges::any_of(c.worlds, [](const auto& w) { return w.world == 5 && w.visits == 1; });
    }));
}

TEST(GumbelSieve, SkipsDecidedRootsWithoutSimulating) {
    for (float extreme : {0.0001f, 0.9999f}) {
        auto s = settings();
        s.simulations = 256;
        s.sieve_gain = 0.06;
        Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
        int queries = 0;
        const auto r = run(search, [&](const Query& q) {
            ++queries;
            auto e = flat(q);
            e.value = extreme;
            e.logits.back() = 3.f;
            return e;
        });
        EXPECT_EQ(queries, 1);
        EXPECT_EQ(r.action, r.actions.back());
    }
}

TEST(GumbelSieve, StopsEarlyWhenPriorHoldsUpAndEscalatesWhenOverturned) {
    auto s = settings();
    s.simulations = 256;
    s.sieve_gain = 0.06;
    auto root_visits = [](const Gumbel& search) {
        int sum = 0;
        for (const auto& edge : root_node(search.graph()).edges) sum += edge.visits;
        return sum;
    };

    Gumbel quiet{stsrl::scenarios::jaw_worm(7), s, 1};
    run(quiet, flat);
    EXPECT_GT(root_visits(quiet), 0);
    EXPECT_LE(root_visits(quiet), 16);

    auto lethal = stsrl::scenarios::jaw_worm(7);
    lethal.monsters.arr[0].curHp = 1;
    lethal.player.curHp = 1;
    const auto actions = legal_actions(lethal);
    const auto loser = std::ranges::find_if(actions, [](Action a) { return a.getActionType() == sts::search::ActionType::END_TURN; });
    ASSERT_NE(loser, actions.end());
    Gumbel urgent{lethal, s, 1};
    const auto r = run(urgent, [&](const Query& q) {
        auto e = flat(q);
        for (std::size_t i = 0; i < q.actions.size(); ++i) e.logits[i] = q.actions[i] == *loser ? 8.f : 0.f;
        return e;
    });
    EXPECT_GT(root_visits(urgent), 16);
    EXPECT_TRUE(wins_now(lethal, r.action));
}

TEST(GumbelSieve, RespectsBudgetCapAndProducesValidPolicy) {
    auto s = settings(1);
    s.simulations = 256;
    s.sieve_gain = 0.02;
    Gumbel search{stsrl::scenarios::jaw_worm(7), s, 1};
    int queries = 0;
    auto answer = noisy();
    const auto r = run(search, [&](const Query& q) { ++queries; return answer(q); });
    EXPECT_LE(queries, s.simulations + 1);
    EXPECT_NEAR(std::accumulate(r.policy.begin(), r.policy.end(), 0.0), 1.0, 1e-5);
    EXPECT_NE(std::ranges::find(r.actions, r.action), r.actions.end());
}
