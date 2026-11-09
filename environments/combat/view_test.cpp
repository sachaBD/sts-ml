#include "environments/combat/view.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <random>
#include <set>

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

TEST(Public, IdenticalCardCopiesInHandAreOneMove) {
    for (const std::uint64_t seed : {1, 2, 3}) {
        auto state = stsrl::scenarios::jaw_worm(seed).battle();
        const auto all = sts::search::Action::getAllActionsInState(state);
        const auto unique = stsrl::combat::legal_actions(state);
        EXPECT_LT(unique.size(), all.size());  // starter hand always holds duplicate Strikes or Defends

        std::set<std::pair<sts::CardId, int>> seen_cards;
        for (const auto a : unique) {
            if (a.getActionType() == sts::search::ActionType::CARD) {
                const auto card = state.cards.hand[a.getSourceIdx()].id;
                EXPECT_TRUE(seen_cards.emplace(card, a.getTargetIdx()).second);
            }
        }
    }
}
