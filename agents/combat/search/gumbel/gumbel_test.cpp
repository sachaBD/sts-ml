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
    EXPECT_TRUE(search.done());
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
            auto evaluate = noisy();
            std::vector<Query> pending;
            while (!search.done()) {
                while (auto query = search.next_query()) pending.push_back(*query);
                if (pending.empty()) { ASSERT_TRUE(search.done()); break; }
                const auto query = pending.back();
                pending.pop_back();
                search.evaluated(query.id, evaluate(query));
            }
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

TEST(Gumbel, UnrescaledValuesScaleSigmaWithoutMinMaxStretching) {
    const auto root = jaw_worm();
    const auto actions = legal_actions(root);
    ASSERT_GE(actions.size(), 2u);
    auto s = settings(0);
    s.simulations = 4;
    s.considered_actions = 2;
    s.rescale_values = false;
    Gumbel search{root, s, 1};
    int queries = 0;
    run(search, [&](const Query& q) {
        ++queries;
        if (queries == 1) {
            std::vector<float> logits(q.actions.size(), -10.f);
            logits[0] = 2.0f;
            logits[1] = 0.0f;
            return Evaluation{0.5f, logits};
        }
        return Evaluation{queries % 2 ? 0.52f : 0.50f, std::vector<float>(q.actions.size(), 0.f)};
    });
    const auto tree = search.tree();
    const auto& edges = tree.nodes[0].edges;
    const double q0 = edges[0].value_sum / edges[0].visits;
    const double q1 = edges[1].value_sum / edges[1].visits;
    const double scale = (s.maxvisit_init + 2) * s.value_scale;
    EXPECT_NEAR(edges[1].sigma - edges[0].sigma, scale * (q1 - q0), 1e-6);
}

TEST(GumbelPipeline, ClaimsRootOnceAndRejectsUnknownOrDuplicateCompletions) {
    auto s = settings();
    s.simulations = 0;
    Gumbel search{jaw_worm(), s, 1};
    EXPECT_FALSE(search.done());
    EXPECT_THROW(search.result(), std::logic_error);
    const auto root = search.next_query();
    ASSERT_TRUE(root);
    EXPECT_FALSE(search.next_query());
    EXPECT_THROW(search.evaluated(root->id + 1, flat(*root)), std::logic_error);
    const auto evaluation = flat(*root);
    search.evaluated(root->id, evaluation);
    EXPECT_THROW(search.evaluated(root->id, evaluation), std::logic_error);
    EXPECT_TRUE(search.done());
    EXPECT_FALSE(search.next_query());
}

TEST(GumbelPipeline, ReadyCompetitorContinuesWhileAnotherEvaluationIsOutstanding) {
    auto s = settings();
    s.simulations = 32;
    s.considered_actions = 2;
    s.max_inflight = 2;
    Gumbel search{jaw_worm(), s, 1};
    const auto root = search.next_query();
    ASSERT_TRUE(root);
    search.evaluated(root->id, flat(*root));
    const auto slow = search.next_query();
    const auto fast = search.next_query();
    ASSERT_TRUE(slow);
    ASSERT_TRUE(fast);
    EXPECT_NE(slow->id, fast->id);
    const auto observation = stsrl::combat::observation_key(slow->state);
    const std::vector<Action> actions(slow->actions.begin(), slow->actions.end());
    EXPECT_FALSE(search.next_query());
    search.evaluated(fast->id, flat(*fast));
    const auto next = search.next_query();
    ASSERT_TRUE(next) << "a ready branch should not wait for its competitor";
    EXPECT_NE(next->id, slow->id);
    EXPECT_EQ(stsrl::combat::observation_key(slow->state), observation);
    EXPECT_TRUE(std::ranges::equal(slow->actions, actions));
    search.evaluated(next->id, flat(*next));
    search.evaluated(slow->id, flat(*slow));
    run(search, flat);
    const auto tree = search.tree();
    EXPECT_EQ(tree.nodes[0].total, s.simulations);
    int visited = 0;
    for (const auto& edge : tree.nodes[0].edges) {
        if (!edge.visits) continue;
        ++visited;
        EXPECT_EQ(edge.visits, s.simulations / s.considered_actions);
    }
    EXPECT_EQ(visited, s.considered_actions);
}

TEST(GumbelPipeline, OutOfOrderCompletionsRespectBoundsAndSimulationAccounting) {
    for (int limit : {1, 2, 4, 16}) {
        for (int simulations : {0, 1, 3, 8, 31, 64}) {
            auto s = settings(1);
            s.max_inflight = limit;
            s.simulations = simulations;
            Gumbel search{jaw_worm(), s, 7};
            std::vector<Query> outstanding;
            std::vector<std::uint64_t> ids;
            int peak = 0;
            while (!search.done()) {
                while (auto q = search.next_query()) {
                    EXPECT_EQ(std::ranges::find(ids, q->id), ids.end());
                    ids.push_back(q->id);
                    outstanding.push_back(*q);
                    EXPECT_LE(outstanding.size(), std::size_t(limit));
                    EXPECT_TRUE(std::ranges::equal(q->actions, legal_actions(q->state)));
                }
                peak = std::max(peak, int(outstanding.size()));
                if (outstanding.empty()) { ASSERT_TRUE(search.done()); break; }
                // Complete the newest first, deliberately leaving older competitors waiting.
                const auto q = outstanding.back();
                outstanding.pop_back();
                search.evaluated(q.id, flat(q));
            }
            EXPECT_LE(ids.size(), std::size_t(simulations + 1));
            if (simulations >= 8 && limit > 1) EXPECT_GT(peak, 1);
            const auto tree = search.tree();
            EXPECT_LE(tree.nodes[0].total, simulations);
            if (simulations > 0) EXPECT_GT(tree.nodes[0].total, 0);
            else EXPECT_EQ(tree.nodes[0].total, 0);
            for (const auto& node : tree.nodes) {
                int visits = 0;
                for (const auto& edge : node.edges) {
                    visits += edge.visits;
                    EXPECT_GE(edge.value_sum, 0);
                    EXPECT_LE(edge.value_sum, edge.visits);
                }
                EXPECT_EQ(visits, node.total);
            }
        }
    }
}

TEST(GumbelPipeline, CompletionsBackUpOnlyTheirOwnBranchAndEliminationWaitsForEveryCompetitor) {
    auto s = settings();
    s.simulations = 8;
    s.considered_actions = 4;
    s.max_inflight = 4;
    Gumbel search{jaw_worm(), s, 1};
    const auto root = search.next_query();
    ASSERT_TRUE(root);
    search.evaluated(root->id, flat(*root));
    std::vector<Query> pending;
    while (auto q = search.next_query()) pending.push_back(*q);
    ASSERT_EQ(pending.size(), std::size_t(s.considered_actions));
    const auto initial = search.tree();
    std::vector<std::size_t> branches;
    for (const auto& q : pending) {
        const auto& edges = initial.nodes[0].edges;
        const auto edge = std::ranges::find_if(edges, [&](const auto& e) {
            return std::ranges::any_of(e.children, [&](const auto& child) { return child.node == int(q.id); });
        });
        ASSERT_NE(edge, edges.end());
        branches.push_back(std::size_t(edge - edges.begin()));
    }
    // Complete all but the strongest competitor; it must still participate in elimination.
    for (std::size_t i = 0; i < pending.size(); ++i) {
        auto evaluation = flat(pending[i]);
        evaluation.value = float(i + 1) / pending.size();
        search.evaluated(pending[i].id, evaluation);
        const auto tree = search.tree();
        EXPECT_EQ(tree.nodes[0].total, int(i + 1));
        for (std::size_t j = 0; j < branches.size(); ++j) {
            const auto& edge = tree.nodes[0].edges[branches[j]];
            EXPECT_EQ(edge.visits, j <= i ? 1 : 0);
            EXPECT_DOUBLE_EQ(edge.value_sum, j <= i ? double(j + 1) / pending.size() : 0);
        }
        if (i + 1 < pending.size()) EXPECT_FALSE(search.next_query());
    }
    run(search, flat);
    const auto tree = search.tree();
    EXPECT_EQ(tree.nodes[0].total, s.simulations);
    for (std::size_t i = 0; i < branches.size(); ++i)
        EXPECT_EQ(tree.nodes[0].edges[branches[i]].visits > 1, i >= branches.size() / 2);
}
