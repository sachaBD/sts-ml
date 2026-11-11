#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/fake_network.hpp"
#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace stsrl::player;
using namespace stsrl::nn;
using stsrl::combat::legal_actions;

namespace {
stsrl::search::SearchFactory search() {
    return stsrl::search::SearchConfig{.sims = 8, .particles = 2, .gumbel_scale = 1, .rollout_mix = 0.5}.factory();
}

// As nn_play decides, given the previous decision's answers; collects what reaches the network.
stsrl::search::Result decide(const Fight& fight, Answers& answers, std::vector<std::vector<float>>& sent) {
    const stsrl::nn::NetworkEvaluator network{Encoding::entity};
    auto random = fight.decision_random();
    Decision decision{search()(fight.state(), random), network, std::move(answers)};
    while (const auto query = decision.pending()) {
        sent.push_back(query->data);
        decision.evaluated(fake_network(*query));
    }
    answers = std::move(decision).answers();
    return decision.result();
}

stsrl::search::Result decide(const Fight& fight) {
    Answers none;
    std::vector<std::vector<float>> sent;
    return decide(fight, none, sent);
}

const std::vector<std::uint64_t> seeds{1, 2, 3};
}  // namespace

TEST(Fight, AutoPlaysForcedMovesAndStopsOnlyAtARealChoice) {
    for (const auto seed : seeds) {
        auto start = stsrl::scenarios::jaw_worm(seed);
        if (seed == 1) start.player.energy = 0;  // begins with a forced End Turn
        Fight fight{start, seed};
        std::mt19937_64 random{seed};
        for (;;) {
            ASSERT_TRUE(fight.over() || legal_actions(fight.state()).size() > 1);
            if (fight.over()) break;
            const auto actions = legal_actions(fight.state());
            fight.play(actions[random() % actions.size()]);
        }
    }
}

// Played with the previous decision's answers, replayed without them.
TEST(Decision, DependsOnlyOnTheFightStartAndTheMovesSoFar) {
    for (const auto seed : seeds) {
        Fight played{stsrl::scenarios::jaw_worm(seed), seed};
        Answers answers;
        std::vector<std::vector<float>> sent;
        std::vector<sts::search::Action> moves;
        std::vector<stsrl::search::Result> results;
        while (!played.over()) {
            results.push_back(decide(played, answers, sent));
            moves.push_back(results.back().action);
            played.play(moves.back());
        }
        for (std::size_t k = 0; k < results.size(); ++k) {
            Fight replay{stsrl::scenarios::jaw_worm(seed), seed};
            for (std::size_t i = 0; i < k; ++i) replay.play(moves[i]);
            const auto r = decide(replay);
            EXPECT_EQ(r.action, results[k].action) << "seed " << seed << " decision " << k;
            EXPECT_EQ(r.policy, results[k].policy) << "seed " << seed << " decision " << k;
            EXPECT_EQ(r.value, results[k].value) << "seed " << seed << " decision " << k;
        }
    }
}

TEST(Decision, NeverResendsARequestThePreviousDecisionSent) {
    int reused = 0;
    for (const auto seed : seeds) {
        Fight fight{stsrl::scenarios::jaw_worm(seed), seed};
        Answers answers;
        std::vector<std::vector<float>> previous;
        while (!fight.over()) {
            std::vector<std::vector<float>> sent, fresh;
            const auto r = decide(fight, answers, sent);
            Answers none;
            decide(fight, none, fresh);
            for (const auto& request : sent) EXPECT_EQ(std::ranges::count(previous, request), 0) << "seed " << seed;
            reused += int(fresh.size() - sent.size());
            previous = std::move(sent);
            fight.play(r.action);
        }
    }
    EXPECT_GT(reused, 0);
}
