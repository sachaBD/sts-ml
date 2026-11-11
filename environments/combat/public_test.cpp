#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <tuple>

using namespace stsrl::combat;
using Task = sts::CardSelectTask;

namespace {
template <typename Pile> std::vector<std::pair<sts::CardId, int>> sorted(const Pile& pile) {
    std::vector<std::pair<sts::CardId, int>> cards;
    for (const auto& card : pile) cards.emplace_back(card.id, card.getUpgradeCount());
    std::ranges::sort(cards);
    return cards;
}

// What playing `action` leads to, ignoring the order of every pile and of the potion slots.
auto result(sts::BattleContext state, Action action) {
    action.execute(state);
    const auto& c = state.cards;
    std::vector<sts::Potion> potions(state.potions.begin(), state.potions.begin() + state.potionCapacity);
    std::ranges::sort(potions);
    return std::tuple{sorted(std::span{c.hand.data(), std::size_t(c.cardsInHand)}), sorted(c.drawPile),
                      sorted(c.discardPile), sorted(c.exhaustPile), potions, state.player.energy, state.player.block,
                      state.monsters.arr[0].curHp, state.monsters.arr[1].curHp};
}

// Jaw Worm and a weaker copy, holding two identical potions; at the start (duplicate Strikes and Defends), or picking from piles of duplicates.
// Duplicates sit at different positions in each pile, so reading the wrong pile or a position is caught.
sts::BattleContext fight(Task task) {
    using enum sts::CardId;
    auto state = stsrl::scenarios::jaw_worm(7);
    state.monsters.arr[1] = state.monsters.arr[0];
    state.monsters.arr[1].idx = 1;
    state.monsters.arr[1].curHp -= 5;
    state.monsters.monsterCount = state.monsters.monstersAlive = 2;
    state.obtainPotion(sts::Potion::FIRE_POTION);
    state.obtainPotion(sts::Potion::FIRE_POTION);
    if (task == Task::INVALID) return state;
    state.inputState = sts::InputState::CARD_SELECT;
    state.cardSelectInfo.cardSelectTask = task;
    state.cardSelectInfo.pickCount = 4;
    const auto fill = [](auto& pile, std::vector<sts::CardInstance> cards) {
        pile.clear();
        for (const auto& card : cards) pile.push_back(card);
    };
    fill(state.cards.discardPile, {{STRIKE_RED}, {BASH}, {STRIKE_RED}, {BASH, true}});
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

// Playing from hand, picking from a pile, and picking several.
INSTANTIATE_TEST_SUITE_P(EveryMechanism, Public, testing::Values(Task::INVALID, Task::HEADBUTT, Task::EXHAUST_MANY),
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
