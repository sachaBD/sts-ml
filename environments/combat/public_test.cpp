#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <tuple>

using namespace stsrl::combat;
using Task = sts::CardSelectTask;

namespace {
using Cards = std::vector<std::pair<sts::CardId, int>>;

template <typename Pile> Cards sorted(const Pile& pile) {
    Cards cards;
    for (const auto& card : pile) cards.emplace_back(card.id, card.getUpgradeCount());
    std::ranges::sort(cards);
    return cards;
}

// What playing `action` leads to, ignoring the order of every pile.
auto result(sts::BattleContext state, Action action) {
    action.execute(state);
    const auto& c = state.cards;
    return std::tuple{sorted(std::span{c.hand.data(), std::size_t(c.cardsInHand)}), sorted(c.drawPile),
                      sorted(c.discardPile), sorted(c.exhaustPile), state.player.energy, state.player.block,
                      state.monsters.arr[0].curHp};
}

// The Jaw Worm start, or a card selection. Every pile holds duplicates, at different positions and of different
// cards in each pile, so a key that reads the wrong pile or a position is caught.
sts::BattleContext fight(Task task) {
    using enum sts::CardId;
    auto state = stsrl::scenarios::jaw_worm(7);
    if (task == Task::INVALID) return state;
    state.inputState = sts::InputState::CARD_SELECT;
    state.cardSelectInfo.cardSelectTask = task;
    const auto fill = [](auto& pile, std::vector<sts::CardInstance> cards) {
        pile.clear();
        for (const auto& card : cards) pile.push_back(card);
    };
    fill(state.cards.drawPile, {{CLEAVE, true}, {ANGER}, {ANGER}, {CLEAVE}});
    state.cardSelectInfo.pickCount = int(state.cards.drawPile.size());  // Scry sees the whole pile
    fill(state.cards.discardPile, {{STRIKE_RED}, {DEFEND_RED}, {STRIKE_RED}, {BASH, true}});
    fill(state.cards.exhaustPile, {{SHRUG_IT_OFF}, {CLOTHESLINE}, {FLEX}, {SHRUG_IT_OFF}});
    const std::vector<sts::CardInstance> hand{{IRON_WAVE}, {POMMEL_STRIKE}, {TWIN_STRIKE}, {POMMEL_STRIKE}};
    std::ranges::copy(hand, state.cards.hand.begin());
    state.cards.cardsInHand = int(hand.size());
    return state;
}

sts::BattleContext reversed(sts::BattleContext state) {
    std::reverse(state.cards.hand.begin(), state.cards.hand.begin() + state.cards.cardsInHand);
    std::ranges::reverse(state.cards.drawPile);
    std::ranges::reverse(state.cards.discardPile);
    std::ranges::reverse(state.cards.exhaustPile);
    return state;
}

class Public : public testing::TestWithParam<Task> {};
}  // namespace

INSTANTIATE_TEST_SUITE_P(EveryPile, Public, testing::Values(Task::INVALID, Task::WARCRY, Task::HEADBUTT, Task::EXHUME,
                                                            Task::SEEK, Task::EXHAUST_MANY, Task::SCRY),
                         [](const auto& info) {
                             return info.param == Task::INVALID ? std::string{"PLAY"} : sts::cardSelectTaskStrings[int(info.param)];
                         });

TEST_P(Public, MovesShareAKeyExactlyWhenTheyDoTheSame) {
    const auto state = fight(GetParam());
    const auto all = Action::getAllActionsInState(state);
    ASSERT_LT(legal_actions(state).size(), all.size());
    for (const auto a : all)
        for (const auto b : all)
            EXPECT_EQ(action_key(state, a) == action_key(state, b), result(state, a) == result(state, b));
}

TEST_P(Public, AKeyDoesTheSameWhateverThePileOrder) {
    const auto state = fight(GetParam());
    const auto other = reversed(state);
    const auto others = legal_actions(other);
    for (const auto a : legal_actions(state)) {
        const auto b = std::ranges::find_if(others, [&](Action o) { return action_key(other, o) == action_key(state, a); });
        ASSERT_NE(b, others.end());
        EXPECT_EQ(result(other, *b), result(state, a));
    }
}
