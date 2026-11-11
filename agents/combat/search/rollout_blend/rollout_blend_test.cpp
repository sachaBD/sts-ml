#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace stsrl::search;

TEST(RolloutBlend, LeafValuesBlendTheEvaluationWithAWonOrLostRollout) {
    for (const auto seed : {1u, 2u})
        for (const float value : {0.2f, 0.7f})
            for (const double mix : {0.0, 0.3, 1.0}) {
                const auto state = stsrl::scenarios::jaw_worm(seed);
                Gumbel::Settings settings;
                settings.simulations = 16;
                settings.particles = 2;
                auto inner = std::make_unique<Gumbel>(state, settings, seed);
                const auto& gumbel = *inner;
                auto search = with_rollouts(std::move(inner), state, mix, seed);
                run(*search, [value](const Query& query) {
                    return Evaluation{value, std::vector<float>(query.actions.size(), 0.f)};
                });
                for (const auto& node : gumbel.graph().nodes) {
                    if (node.terminal) continue;
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
