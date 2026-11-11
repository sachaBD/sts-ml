// Action keys name what a move chooses, never where it sits.
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <string>
#include <tuple>

using namespace stsrl::combat;
using Task = sts::CardSelectTask;

namespace {
// What a move does, written out: its kind, its target, and the cards it chooses (by id and upgrades).
using Choice = std::tuple<sts::search::ActionType, int, std::vector<std::pair<sts::CardId, int>>>;

Choice choice(const sts::BattleContext& state, Action action) {
    using enum sts::search::ActionType;
    const auto type = action.getActionType();
    std::vector<int> positions;
    if (type == CARD) positions.push_back(action.getSourceIdx());
    if (type == SINGLE_CARD_SELECT) positions.push_back(action.getSelectIdx());
    if (type == MULTI_CARD_SELECT) for (const int i : action.getSelectedIdxs()) positions.push_back(i);
    const auto pile = type == CARD ? std::span<const sts::CardInstance>{state.cards.hand.data(), std::size_t(state.cards.cardsInHand)}
                                   : cards_in(state, selection_pile(state));
    Choice c{type, type == CARD ? action.getTargetIdx() : -1, {}};
    for (const int i : positions) std::get<2>(c).emplace_back(pile[i].id, pile[i].getUpgradeCount());
    std::ranges::sort(std::get<2>(c));
    return c;
}

// The Jaw Worm start (a hand of duplicate Strikes and Defends), or a card selection among duplicates in every pile.
sts::BattleContext fight(Task task) {
    auto state = stsrl::scenarios::jaw_worm(7);
    if (task == Task::INVALID) return state;
    const std::vector<sts::CardInstance> cards{
        {sts::CardId::STRIKE_RED}, {sts::CardId::DEFEND_RED}, {sts::CardId::STRIKE_RED},
        {sts::CardId::BASH, true}, {sts::CardId::DEFEND_RED}};
    state.inputState = sts::InputState::CARD_SELECT;
    state.cardSelectInfo.cardSelectTask = task;
    state.cardSelectInfo.pickCount = int(cards.size());
    for (auto* pile : {&state.cards.drawPile, &state.cards.discardPile, &state.cards.exhaustPile}) {
        pile->clear();
        for (const auto& card : cards) pile->push_back(card);
    }
    std::ranges::copy(cards, state.cards.hand.begin());
    state.cards.cardsInHand = int(cards.size());
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

// Hand play, hand pick, discard, exhaust and draw picks, and multi-selects from hand and draw.
INSTANTIATE_TEST_SUITE_P(EveryPile, Public, testing::Values(Task::INVALID, Task::WARCRY, Task::HEADBUTT, Task::EXHUME,
                                                            Task::SEEK, Task::EXHAUST_MANY, Task::SCRY),
                         [](const auto& info) {
                             return info.param == Task::INVALID ? std::string{"PLAY"} : sts::cardSelectTaskStrings[int(info.param)];
                         });

TEST_P(Public, OneMovePerChoice) {
    const auto state = fight(GetParam());
    std::set<Choice> all, legal;
    for (const auto a : Action::getAllActionsInState(state)) all.insert(choice(state, a));
    for (const auto a : legal_actions(state)) EXPECT_TRUE(legal.insert(choice(state, a)).second);
    EXPECT_LT(all.size(), Action::getAllActionsInState(state).size());  // the fixture has duplicates to merge
    EXPECT_EQ(legal, all);
}

TEST_P(Public, AKeyChoosesTheSameCardsInAnyPileOrder) {
    const auto state = fight(GetParam());
    const auto other = reversed(state);
    const auto others = legal_actions(other);
    for (const auto a : legal_actions(state)) {
        const auto b = std::ranges::find_if(others, [&](Action o) { return action_key(other, o) == action_key(state, a); });
        ASSERT_NE(b, others.end());
        EXPECT_EQ(choice(other, *b), choice(state, a));
    }
}
