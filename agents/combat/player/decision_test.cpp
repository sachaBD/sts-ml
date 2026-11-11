#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/fake_network.hpp"
#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <map>

using namespace stsrl::player;
using namespace stsrl::nn;
using stsrl::combat::legal_actions;

namespace {
stsrl::search::SearchFactory search() {
    return stsrl::search::SearchConfig{.sims = 8, .particles = 2, .gumbel_scale = 1, .rollout_mix = 0.5}.factory();
}

// As nn_play decides.
stsrl::search::Result decide(const Fight& fight) {
    const stsrl::nn::NetworkEvaluator network{Encoding::entity};
    auto random = fight.decision_random();
    Decision decision{search()(fight.state(), random), network};
    while (const auto query = decision.pending()) decision.evaluated(fake_network(*query));
    return decision.result();
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

TEST(Decision, DependsOnlyOnTheFightStartAndTheMovesSoFar) {
    for (const auto seed : seeds) {
        Fight played{stsrl::scenarios::jaw_worm(seed), seed};
        std::vector<sts::search::Action> moves;
        std::vector<stsrl::search::Result> results;
        while (!played.over()) {
            results.push_back(decide(played));
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


TEST(Decision, InspectionPreservesNetworkAndRolloutValueSemantics) {
    for (const auto encoding : {Encoding::entity, Encoding::inventory_v1})
        for (const double mix : {0.0, 0.3, 1.0}) {
            Fight fight{stsrl::scenarios::jaw_worm(7), 7};
            const NetworkEvaluator network{encoding};
            auto random = fight.decision_random();
            const stsrl::search::SearchConfig config{16, 2, 1, mix};
            auto search = config.create(fight.state(), random);
            Decision decision{std::move(search.search), network};
            std::map<std::uint64_t, float> predictions;
            while (const auto tokens = decision.pending()) {
                const auto prediction = fake_network(*tokens);
                predictions.emplace(stsrl::combat::observation_key(search.gumbel.pending()->state), prediction.value);
                decision.evaluated(prediction);
            }
            const auto graph = search.gumbel.graph();
            for (const auto& node : graph.nodes) {
                if (node.terminal) continue;
                const auto value = predictions.at(node.observation);
                if (mix == 0) {
                    EXPECT_EQ(node.evaluation_value, value);
                } else {
                    const double rollout = (node.evaluation_value - (1 - mix) * value) / mix;
                    EXPECT_NEAR(rollout, std::round(rollout), 1e-6);
                    EXPECT_TRUE(std::round(rollout) == 0 || std::round(rollout) == 1);
                }
            }
        }
}
