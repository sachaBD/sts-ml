#include "agents/combat/search/filter/filter.hpp"
#include "agents/combat/search/filter/belief.hpp"
#include "agents/combat/search/filter/tree.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <set>

using namespace stsrl;

TEST(Filter, RepeatedConsistentEvidenceReducesValueOfAnotherMeasurement) {
    std::vector<double> ys;
    double prev = search::filter::kalman(0.5, 0.0025, ys).innovation;
    for (int i = 0; i < 16; ++i) {
        ys.push_back(0.65);
        const auto b = search::filter::kalman(0.5, 0.0025, ys);
        EXPECT_LT(b.innovation, prev);
        EXPECT_GT(b.mean, 0.5);
        EXPECT_LT(b.mean, 0.65);
        prev = b.innovation;
    }
}

TEST(Filter, CapAppliesToNewRequestsAndPriceOnlyTruncatesTrace) {
    auto state = scenarios::jaw_worm(19);
    auto trace = [&](int cap, double price) {
        auto tree = search::filter_search(state, cap, 8, price, search::FilterMode::full, 17);
        std::vector<std::uint64_t> keys;
        auto result = search::run(*tree, [&](const search::Query& q) {
            auto k = combat::observation_key(q.state);
            keys.push_back(k);
            return search::Evaluation{
                float(.3 + (k % 400) / 1000.), std::vector<float>(q.actions.size(), 0), {.01, .06, .07, .08}};
        });
        EXPECT_LE(keys.size(), std::size_t(cap + 1));
        EXPECT_EQ(std::set(keys.begin(), keys.end()).size(), keys.size());
        EXPECT_NE(std::ranges::find(result.actions, result.action), result.actions.end());
        return keys;
    };
    auto all = trace(64, 0.);
    auto paid = trace(64, .01);
    auto stop = trace(64, 1.);
    EXPECT_EQ(stop.size(), 1u);
    ASSERT_GT(paid.size(), 1u);
    ASSERT_LE(paid.size(), all.size());
    EXPECT_TRUE(std::equal(paid.begin(), paid.end(), all.begin()));
    trace(1, 0.);
}

TEST(Filter, SearchCrossesIntoNextTurnAndBeyondWithoutTurnLimit) {
    auto state = scenarios::jaw_worm(19);
    const int start_turn = state.turn;
    search::FilterTrace trace;
    auto tree = search::filter_search(state, 80, 2, 0., search::FilterMode::full, 17, &trace);
    search::run(*tree, [&](const search::Query& q) {
        // Favor End turn so the best path quickly advances across multiple turns.
        std::vector<float> logits(q.actions.size(), -10.f);
        for (std::size_t i = 0; i < q.actions.size(); ++i)
            if (q.actions[i].bits == 0)  // end turn or first action fallback
                logits[i] = 0.f;
        logits.back() = 0.f;
        return search::Evaluation{0.5f, logits, {.02, .05, .07, .09}};
    });
    int max_turn = start_turn;
    for (const auto& n : trace.nodes)
        max_turn = std::max(max_turn, n.turn);
    EXPECT_GE(max_turn, start_turn + 2);
}

TEST(Filter, DrawPileSelectionPlaysThePublicCardOnEachParticle) {
    using enum sts::CardId;
    auto state = scenarios::jaw_worm(7);
    state.inputState = sts::InputState::CARD_SELECT;
    state.cardSelectInfo.cardSelectTask = sts::CardSelectTask::SECRET_WEAPON;
    state.cards.drawPile.clear();
    for (auto id : {BASH, STRIKE_RED, DEFEND_RED})
        state.cards.drawPile.push_back({id});
    for (int seed = 0; seed < 12; ++seed) {
        auto tree = search::filter_search(state, 8, 1, 0., search::FilterMode::root_only, seed);
        const auto root = tree->pending().front();
        ASSERT_EQ(root.actions.size(), 2u);
        const std::vector<search::Action> moves(root.actions.begin(), root.actions.end());
        const search::Evaluation answer{.5f, {0.f, 0.f}, {.01, .02, .03, .04}};
        tree->evaluated(std::span{&answer, 1});
        for (int move_idx = 0; move_idx < 2; ++move_idx) {
            const auto pending = tree->pending();
            ASSERT_EQ(pending.size(), 1u);
            const auto& cards = pending[0].state.cards;
            ASSERT_EQ(cards.cardsInHand, state.cards.cardsInHand + 1);
            EXPECT_EQ(cards.hand[cards.cardsInHand - 1].id, state.cards.drawPile[moves[move_idx].getSelectIdx()].id);
            const search::Evaluation child_eval{.5f, std::vector<float>(pending[0].actions.size(), 0.f), {0, 0, 0, 0}};
            tree->evaluated(std::span{&child_eval, 1});
        }
    }
}

TEST(Filter, InspectionRecordsCompletedStepsWithoutChangingSearch) {
    const auto state = scenarios::jaw_worm(19);
    auto replay = [&](search::FilterTrace* trace) {
        auto tree = search::filter_search(state, 64, 8, .001, search::FilterMode::full, 17, trace);
        std::vector<std::uint64_t> queries;
        auto result = search::run(*tree, [&](const search::Query& q) {
            const auto key = combat::observation_key(q.state);
            queries.push_back(key);
            return search::Evaluation{float(.3 + (key % 400) / 1000.), std::vector<float>(q.actions.size()),
                                      {.01, .06, .07, .08}};
        });
        if (trace) {
            EXPECT_FALSE(trace->steps.empty());
            EXPECT_EQ(trace->steps.size(), std::size_t(result.paired_root_trials + result.paired_interior_trials));
            EXPECT_LE(trace->nodes.size(), trace->steps.size() + 1u);
            EXPECT_EQ(trace->spent + 1, int(queries.size()));
            EXPECT_FALSE(trace->stop.empty());
            for (const auto& n : trace->nodes)
                if (!n.terminal)
                    EXPECT_EQ(n.thinking.size(), 4u);
        }
        return std::pair{queries, result.action.bits};
    };
    search::FilterTrace trace;
    EXPECT_EQ(replay(nullptr), replay(&trace));
}

TEST(Filter, DecidedStatesHaveZeroUncertaintyAndExitInOneEval) {
    const std::array<float, 4> noisy{0.01f, 0.02f, 0.03f, 0.04f};
    EXPECT_EQ(search::filter::prior_sd(1.0, noisy), 0.0);
    EXPECT_EQ(search::filter::prior_sd(0.0, noisy), 0.0);
    auto state = scenarios::jaw_worm(19);
    auto tree = search::filter_search(state, 32, 8, 0.0005, search::FilterMode::full, 17);
    int evals = 0;
    search::run(*tree, [&](const search::Query& q) {
        ++evals;
        return search::Evaluation{1.0f, std::vector<float>(q.actions.size(), 0.f), {0.01f, 0.02f, 0.03f, 0.04f}};
    });
    EXPECT_EQ(evals, 1);
}

TEST(Filter, WorldOffsetsCancelSharedDrawLuckWithoutErasingUniformShift) {
    search::filter::Tree tree;
    auto state = scenarios::jaw_worm(19);
    const search::filter::Tree::EvalSummary root_ev{0.5, 0.1, {1, 2}, {0.f, 0.f}, {0, 1}};
    tree.add_root(100, state, root_ev);
    auto leaf_ev = [](double v) {
        return search::filter::Tree::EvalSummary{v, 0.01, {9}, {0.f}, {0}};
    };
    auto turn2 = state;
    turn2.turn = state.turn + 1;
    // Both arms see W0 (unlucky -0.3) and W1 (lucky +0.3), with arm 1 genuinely +0.05 better.
    const auto a0w0 = leaf_ev(0.20), a0w1 = leaf_ev(0.80), a1w0 = leaf_ev(0.25), a1w1 = leaf_ev(0.85);
    tree.add_child(0, 0, 0, true, 101, turn2, &a0w0);
    tree.add_child(0, 0, 1, true, 102, turn2, &a0w1);
    tree.add_child(0, 1, 0, true, 103, turn2, &a1w0);
    tree.add_child(0, 1, 1, true, 104, turn2, &a1w1);
    tree.backup(false);
    // Shared luck cancels even when arm 1 has only 1 world or both have 2 worlds: arm 1 leads by +5 pp.
    EXPECT_GT(tree.node(0).edges[1].q, tree.node(0).edges[0].q + 0.03);
    EXPECT_LT(tree.node(0).edges[1].innovation, 0.05);

    // Deeper lookahead on the prior line itself (raising its W0 Turn+1 child backed-up value) still gives prior credit.
    const auto better_turn2 = leaf_ev(0.35);
    tree.add_child(1, 0, -1, false, 105, turn2, &better_turn2);
    tree.backup(false);
    EXPECT_GT(tree.node(0).edges[0].q, 0.52);
}

TEST(Filter, RareLowPolicyMoveCannotHijackRootFromOneNoisyEval) {
    search::filter::Tree tree;
    auto state = scenarios::jaw_worm(19);
    // Arm 0 has 90%+ policy (logit 0), Arm 1 has ~0.8% policy (logit -4.7 -> prior 14 pp lower).
    const search::filter::Tree::EvalSummary root_ev{0.70, 0.02, {1, 2}, {0.f, -4.7f}, {0, 1}};
    tree.add_root(300, state, root_ev);
    const search::filter::Tree::EvalSummary main_child{0.69, 0.01, {9}, {0.f}, {0}};
    const search::filter::Tree::EvalSummary noisy_ood{0.75, 0.01, {9}, {0.f}, {0}};
    tree.add_child(0, 0, -1, false, 301, state, &main_child);
    tree.add_child(0, 1, -1, false, 302, state, &noisy_ood);
    tree.backup(false);
    // Kalman shrinkage against the policy prior keeps the 90% move ahead of one +5pp noisy spike on a 0.8% move.
    EXPECT_EQ(tree.leader(0), 0);
}

TEST(Filter, SharedTranspositionHasZeroRootVoi) {
    search::filter::Tree tree;
    auto state = scenarios::jaw_worm(19);
    const search::filter::Tree::EvalSummary root_ev{0.50, 0.05, {1, 2}, {0.f, 0.f}, {0, 1}};
    tree.add_root(100, state, root_ev);
    // Both root arms transpose into the exact same public state (key 999) with high uncertainty.
    const search::filter::Tree::EvalSummary shared{0.50, 0.20, {3, 4}, {0.f, 0.f}, {0, 1}};
    tree.add_child(0, 0, -1, false, 999, state, &shared);
    tree.add_child(0, 1, -1, false, 999, state, &shared);
    tree.backup(false);
    // Because expanding key 999 shifts both root arms equally, best_candidate skips it (nullopt).
    EXPECT_FALSE(tree.best_candidate(8, false).has_value());
}

TEST(Filter, InteriorPreChanceBranchKeepsCumulativePolicyPrior) {
    search::filter::Tree tree;
    auto state = scenarios::jaw_worm(19);
    auto turn2 = state;
    turn2.turn = state.turn + 1;
    const search::filter::Tree::EvalSummary root_ev{0.50, 0.05, {1, 2}, {0.f, -0.5f}, {0, 1}};
    tree.add_root(100, state, root_ev);
    // Root arm 1 (logit -0.5) branches deterministically to c1, then takes a rare interior move (logit -4.0) to c2.
    const search::filter::Tree::EvalSummary c1_ev{0.50, 0.05, {3, 4}, {0.f, -4.0f}, {0, 1}};
    const search::filter::Tree::EvalSummary c2_ev{0.50, 0.05, {5}, {0.f}, {0}};
    const search::filter::Tree::EvalSummary noisy{0.55, 0.01, {6}, {0.f}, {0}};
    const auto [c1, ok1] = tree.add_child(0, 1, -1, false, 101, state, &c1_ev);
    const auto [c2, ok2] = tree.add_child(c1, 1, -1, false, 102, state, &c2_ev);
    ASSERT_TRUE(ok1 && ok2);
    tree.add_child(c2, 0, 0, true, 103, turn2, &noisy);
    tree.backup(false);
    // Cumulative prior = 0.50 + 0.03 * (-0.5 - 4.0) = 0.365, so one +0.55 eval stays well below 0.50.
    EXPECT_LT(tree.node(c2).edges[0].q, 0.48);
}

TEST(Filter, UnpairedDeepChanceEdgeRegularizesAgainstDrawNoise) {
    search::filter::Tree tree;
    auto state = scenarios::jaw_worm(19);
    auto turn2 = state;
    turn2.turn = state.turn + 1;
    auto turn3 = state;
    turn3.turn = state.turn + 2;
    const search::filter::Tree::EvalSummary root_ev{0.50, 0.05, {1, 2}, {0.f, 0.f}, {0, 1}};
    tree.add_root(100, state, root_ev);
    // Turn-2 child inside W0 with tiny end-of-turn prior_sd = 0.005 rolls one lucky unpaired Turn-3 draw (+20 pp).
    const search::filter::Tree::EvalSummary t2_ev{0.50, 0.005, {3}, {0.f}, {0}};
    const search::filter::Tree::EvalSummary t3_lucky{0.70, 0.01, {4}, {0.f}, {0}};
    const auto [t2, ok] = tree.add_child(0, 0, 0, true, 101, turn2, &t2_ev);
    ASSERT_TRUE(ok);
    tree.add_child(t2, 0, 0, true, 102, turn3, &t3_lucky);
    tree.backup(false);
    // Unpaired Bernoulli draw variance floor (0.05 SD) prevents a tiny-sd node from jumping 50% (+10 pp) on 1 draw.
    EXPECT_LT(tree.node(t2).edges[0].q, 0.52);
}

TEST(Filter, ExploreSamplesThePlayedMoveFromThePolicyAndZeroPlaysTheLeader) {
    auto state = scenarios::jaw_worm(19);
    auto play = [&](double explore, std::uint64_t seed) {
        auto tree = search::filter_search(state, 0, 8, 0., search::FilterMode::full, 17, nullptr, explore, seed);
        return search::run(*tree, [](const search::Query& q) {
            std::vector<float> logits;
            for (std::size_t i = 0; i < q.actions.size(); ++i)
                logits.push_back(-.7f * float(i));
            return search::Evaluation{.5f, logits, {.01, .06, .07, .08}};
        });
    };
    const auto first = play(0, 0);
    const int n = int(first.actions.size());
    ASSERT_GE(n, 2);
    double z = 0;
    for (int i = 0; i < n; ++i)
        z += std::exp(-.7 * i);
    std::vector<int> counts(n);
    constexpr int draws = 4000;
    for (std::uint64_t seed = 0; seed < draws; ++seed) {
        const auto r = play(1, seed);
        ++counts[std::ranges::find(r.actions, r.action) - r.actions.begin()];
        EXPECT_EQ(play(0, seed).action, first.actions[0]);
    }
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(double(counts[i]) / draws, std::exp(-.7 * i) / z, .03) << i;
}
