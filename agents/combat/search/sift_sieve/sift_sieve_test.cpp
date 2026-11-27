#include "agents/combat/search/sift_sieve/sift_sieve.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace stsrl::search;

TEST(SiftSieve, OneRootEvalAndGreedyTiesUseLegalActionOrder) {
    const auto state = stsrl::scenarios::jaw_worm(7);
    const auto actions = stsrl::combat::legal_actions(state);
    ASSERT_GE(actions.size(), 3u);
    std::mt19937_64 random{1};
    auto search = sift_sieve(RootChoice::argmax)(state, 0, random);
    int evals = 0;
    const auto result = run(*search, [&](const Query& q) {
        ++evals;
        EXPECT_TRUE(std::ranges::equal(q.actions, actions));
        Evaluation answer{.625f, std::vector<float>(actions.size(), 999.f)};
        answer.logits[1] = answer.logits[2] = 1000.f;
        return answer;
    });
    EXPECT_EQ(evals, 1);
    EXPECT_EQ(result.action, actions[1]);
    EXPECT_EQ(result.value, .625f);
    EXPECT_NEAR(result.policy[1] / result.policy[0], std::exp(1.), 1e-6);
}

TEST(SiftSieve, SeededSamplingFollowsTheReportedPolicy) {
    const auto state = stsrl::scenarios::jaw_worm(7);
    auto factory = sift_sieve(RootChoice::sample);
    std::mt19937_64 random{1}, repeat{1};
    constexpr int draws = 4096;
    int selected = 0;
    double p = 0;
    auto evaluate = [](const Query& q) {
        Evaluation answer{.5f, std::vector<float>(q.actions.size(), 0.f)};
        answer.logits.front() = 1;
        return answer;
    };
    for (int i = 0; i < draws; ++i) {
        const auto result = run(*factory(state, 0, random), evaluate);
        EXPECT_EQ(result.action, run(*factory(state, 0, repeat), evaluate).action);
        selected += result.action == result.actions.front();
        p = result.policy.front();
    }
    EXPECT_NEAR(selected, draws * p, 6 * std::sqrt(draws * p * (1 - p)));
}
