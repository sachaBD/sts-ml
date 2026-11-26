#include "agents/combat/search/paired_turn/paired_turn.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <unordered_set>

using namespace stsrl;
using namespace stsrl::search;

namespace {
Evaluation end_turn_policy(const Query& query, float value) {
    std::vector<float> logits(query.actions.size());
    for (std::size_t i = 0; i < logits.size(); ++i)
        if (query.actions[i].getActionType() == sts::search::ActionType::END_TURN) logits[i] = 1;
    return {value, std::move(logits)};
}
}  // namespace

TEST(PairedTurn, ZeroBudgetUsesOnlyTheRootPolicy) {
    auto state = scenarios::jaw_worm(7);
    std::mt19937_64 rng{42};
    auto search = paired_turn(4, 0)(state, 0, rng);
    int queries = 0;
    const auto result = run(*search, [&](const Query& query) {
        ++queries;
        return end_turn_policy(query, 0.4f);
    });
    EXPECT_EQ(queries, 1);
    EXPECT_EQ(result.action.getActionType(), sts::search::ActionType::END_TURN);
    EXPECT_FLOAT_EQ(result.value, 0.4f);
    EXPECT_NEAR(std::accumulate(result.policy.begin(), result.policy.end(), 0.0), 1, 1e-6);
}

TEST(PairedTurn, SpendsTightBudgetOnCompleteWorldsAndReadsAtTurnBoundary) {
    for (const auto seed : {7, 11, 23}) {
        SCOPED_TRACE(seed);
        const auto state = scenarios::jaw_worm(seed);
        const auto actions = combat::legal_actions(state);
        ASSERT_GE(actions.size(), 3u);
        const auto worlds = combat::particles(state, 4, state.seed ^ (std::uint64_t(state.turn) << 32));
        std::unordered_set<std::uint64_t> expected;
        // Seven lanes can buy two complete three-way worlds, but not a third world or a paired rerun.
        for (const auto& [w, i] : {std::pair{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}) {
            auto world = worlds[w];
            actions[i].execute(world);
            while (world.turn == state.turn && world.outcome == sts::Outcome::UNDECIDED) {
                const auto next = combat::legal_actions(world);
                auto end = std::ranges::find_if(next, [](Action a) {
                    return a.getActionType() == sts::search::ActionType::END_TURN;
                });
                ASSERT_TRUE(end != next.end() || next.size() == 1);
                (end != next.end() ? *end : next.front()).execute(world);
            }
            if (world.outcome == sts::Outcome::UNDECIDED) expected.insert(combat::observation_key(world));
        }
        ASSERT_FALSE(expected.empty());
        std::mt19937_64 rng{42};
        auto search = paired_turn(4, 0)(state, 7, rng);
        const auto root = combat::observation_key(state);
        std::unordered_set<std::uint64_t> seen, boundaries;
        const auto result = run(*search, [&](const Query& query) {
            const auto key = combat::observation_key(query.state);
            EXPECT_TRUE(seen.insert(key).second);
            if (key == root) return Evaluation{0.5f, std::vector<float>(actions.size())};
            if (query.state.turn > state.turn) {
                EXPECT_EQ(query.state.turn, state.turn + 1);
                boundaries.insert(key);
            }
            return end_turn_policy(query, 0.65f);
        });
        EXPECT_EQ(boundaries, expected);
        EXPECT_FLOAT_EQ(result.value, 0.65f);
    }
}

TEST(PairedTurn, RefinementBlendsTheTwoBoundaries) {
    const auto state = scenarios::jaw_worm(7);
    std::mt19937_64 rng{42};
    auto search = paired_turn(6, 0)(state, 18, rng);
    int second_turn = 0;
    const auto result = run(*search, [&](const Query& query) {
        if (query.state.turn > state.turn + 1) ++second_turn;
        const auto value = query.state.turn == state.turn + 1 ? 0.8f : 0.2f;
        if (query.state.turn == state.turn && combat::observation_key(query.state) == combat::observation_key(state))
            return Evaluation{value, std::vector<float>(query.actions.size())};
        return end_turn_policy(query, value);
    });
    EXPECT_GT(second_turn, 0);
    EXPECT_FLOAT_EQ(result.value, 0.5f);
}

TEST(PairedTurn, WeakScreenGivesAnUnscreenedMoveATwoTurnComparison) {
    const auto state = scenarios::jaw_worm(7);
    const auto actions = combat::legal_actions(state);
    ASSERT_GE(actions.size(), 4u);
    const auto end = std::ranges::find_if(actions, [](Action a) {
        return a.getActionType() == sts::search::ActionType::END_TURN;
    });
    ASSERT_GE(end - actions.begin(), 3);
    ASSERT_NE(end, actions.end());
    const auto worlds = combat::particles(state, 4, state.seed ^ (std::uint64_t(state.turn) << 32));
    std::unordered_set<std::uint64_t> good;
    for (auto world : worlds) {
        end->execute(world);
        ASSERT_EQ(world.outcome, sts::Outcome::UNDECIDED);
        good.insert(combat::observation_key(world));
        const auto next = combat::legal_actions(world);
        const auto next_end = std::ranges::find_if(next, [](Action a) {
            return a.getActionType() == sts::search::ActionType::END_TURN;
        });
        ASSERT_NE(next_end, next.end());
        next_end->execute(world);
        ASSERT_EQ(world.outcome, sts::Outcome::UNDECIDED);
        good.insert(combat::observation_key(world));
    }
    std::mt19937_64 rng{42};
    auto search = paired_turn(4, 0)(state, 18, rng);
    const auto root = combat::observation_key(state);
    std::size_t queries = 0;
    const auto result = run(*search, [&](const Query& query) {
        const auto obs = combat::observation_key(query.state);
        if (obs == root) {
            std::vector<float> prior(actions.size(), -8.0f);
            std::fill_n(prior.begin(), 3, 0.0f);
            prior[end - actions.begin()] = -0.25f;
            return Evaluation{0.8f, std::move(prior)};
        }
        if (good.contains(obs)) ++queries;
        return end_turn_policy(query, good.contains(obs) ? 0.9f : 0.3f);
    });
    EXPECT_EQ(result.action, *end);
    EXPECT_EQ(queries, good.size());
    EXPECT_FLOAT_EQ(result.value, 0.9f);
}
