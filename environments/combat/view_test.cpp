#include "environments/combat/view.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <random>

TEST(View, IsTheSameInEveryWorldThePlayerCannotTellApart) {
    for (const std::uint64_t seed : {1, 2, 3}) {
        auto state = stsrl::scenarios::jaw_worm(seed).battle();
        std::mt19937_64 random{seed};
        while (state.outcome == sts::Outcome::UNDECIDED) {
            const auto seen = stsrl::combat::view(state);
            for (const auto& world : stsrl::combat::particles(state, 4)) ASSERT_EQ(stsrl::combat::view(world), seen);
            const auto actions = stsrl::combat::legal_actions(state);
            actions[random() % actions.size()].execute(state);
        }
    }
}
