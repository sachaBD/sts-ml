#include "agents/combat/search/filter/filter.hpp"
#include "agents/combat/search/filter/belief.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <set>

using namespace stsrl;
TEST(Filter, RepeatedConsistentEvidenceReducesValueOfAnotherMeasurement) {
    search::filter::Gap g;
    for (int i = 0; i < 16; ++i) {
        const double before = g.innovation();
        g.observe(.15);
        EXPECT_LT(g.innovation(), before);
        EXPECT_GT(g.mean(), 0.);
        EXPECT_LT(g.mean(), .15);
    }
}
TEST(Filter, SharedInteriorRevisionCannotChangeRootOrdering) {
    for (double m : {-.1, 0., .1})
        EXPECT_EQ(search::filter::branch_option(-.02, 0., m, .1), 0.);
    EXPECT_GT(search::filter::branch_option(-.02, 1., 0., .1), 0.);
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
    auto paid = trace(64, .001);
    auto stop = trace(64, 1.);
    EXPECT_EQ(stop.size(), 1u);
    ASSERT_GT(paid.size(), 1u);
    ASSERT_LE(paid.size(), all.size());
    EXPECT_TRUE(std::equal(paid.begin(), paid.end(), all.begin()));
    trace(1, 0.);
}

TEST(Filter, BranchPriceAgreesWithNormalIntegralForEitherDirectionOfReach) {
    for (double d : {-.8, .6})
        for (double offset : {-.1, .04}) {
            constexpr double mean = -.025, sd = .15;
            double integrated = 0;
            for (int i = 0; i < 16000; ++i) {
                const double z = -8. + (i + .5) * .001;
                integrated += .001 * std::exp(-.5 * z * z) / std::sqrt(2. * std::numbers::pi) *
                              std::max(0., offset + d * std::max(0., mean + sd * z));
            }
            EXPECT_NEAR(search::filter::branch_option(offset, d, mean, sd), integrated, 1e-8);
        }
}

TEST(Filter, DrawPileSelectionPlaysThePublicCardOnEachParticle) {
    using enum sts::CardId;
    auto state = scenarios::jaw_worm(7);
    state.inputState = sts::InputState::CARD_SELECT;
    state.cardSelectInfo.cardSelectTask = sts::CardSelectTask::SECRET_WEAPON;
    state.cards.drawPile.clear();
    for (auto id : {BASH, STRIKE_RED, DEFEND_RED}) state.cards.drawPile.push_back({id});
    for (int seed = 0; seed < 12; ++seed) {
        auto tree = search::filter_search(state, 8, 1, 0., search::FilterMode::full, seed);
        const auto root = tree->pending().front();
        ASSERT_EQ(root.actions.size(), 2u);
        const std::vector<search::Action> moves(root.actions.begin(), root.actions.end());
        const search::Evaluation answer{.5f, {0.f, 0.f}, {0, 0, 0, 0}};
        tree->evaluated(std::span{&answer, 1});
        const auto pair = tree->pending();
        ASSERT_EQ(pair.size(), 2u);
        for (int lane = 0; lane < 2; ++lane) {
            const auto& cards = pair[lane].state.cards;
            ASSERT_EQ(cards.cardsInHand, state.cards.cardsInHand + 1);
            EXPECT_EQ(cards.hand[cards.cardsInHand - 1].id, state.cards.drawPile[moves[1-lane].getSelectIdx()].id);
        }
    }
}

TEST(Filter, InspectionRecordsCompletedTrialsWithoutChangingSearch) {
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
            EXPECT_FALSE(trace->trials.empty());
            EXPECT_EQ(trace->trials.size(), result.paired_root_trials + result.paired_interior_trials);
            EXPECT_EQ(trace->spent + 1, queries.size());
            EXPECT_FALSE(trace->stop.empty());
        }
        return std::pair{queries, result.action.bits};
    };
    search::FilterTrace trace;
    EXPECT_EQ(replay(nullptr), replay(&trace));
}
TEST(Filter, ExploreSamplesThePlayedMoveFromThePolicyAndZeroPlaysTheLeader) {
    auto state = scenarios::jaw_worm(19);
    auto play = [&](double explore, std::uint64_t seed) {
        auto tree = search::filter_search(state, 0, 8, 0., search::FilterMode::full, 17, nullptr, explore, seed);
        return search::run(*tree, [](const search::Query& q) {
            std::vector<float> logits;
            for (std::size_t i = 0; i < q.actions.size(); ++i) logits.push_back(-.7f * float(i));
            return search::Evaluation{.5f, logits, {.01, .06, .07, .08}};
        });
    };
    const auto first = play(0, 0);
    const int n = int(first.actions.size());
    ASSERT_GE(n, 2);
    double z = 0;
    for (int i = 0; i < n; ++i) z += std::exp(-.7 * i);
    std::vector<int> counts(n);
    constexpr int draws = 4000;
    for (std::uint64_t seed = 0; seed < draws; ++seed) {
        const auto r = play(1, seed);
        ++counts[std::ranges::find(r.actions, r.action) - r.actions.begin()];
        EXPECT_EQ(play(0, seed).action, first.actions[0]);
    }
    for (int i = 0; i < n; ++i) EXPECT_NEAR(double(counts[i]) / draws, std::exp(-.7 * i) / z, .03) << i;
}
