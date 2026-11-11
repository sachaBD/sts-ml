#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>

using namespace stsrl::combat;
using Type = sts::search::ActionType;
using enum sts::CardId;

namespace {
sts::BattleContext with_hand(std::vector<sts::CardInstance> cards) {
    sts::BattleContext state;
    std::ranges::copy(cards, state.cards.hand.begin());
    state.cards.cardsInHand = int(cards.size());
    return state;
}

sts::BattleContext headbutt(std::vector<sts::CardInstance> discard) {
    sts::BattleContext state;
    state.inputState = sts::InputState::CARD_SELECT;
    state.cardSelectInfo.cardSelectTask = sts::CardSelectTask::HEADBUTT;
    for (const auto& card : discard) state.cards.discardPile.push_back(card);
    return state;
}

// A whole, initialised state whose hand, discard and exhaust piles all hold `cards`.
sts::BattleContext with_piles(std::vector<sts::CardInstance> cards) {
    auto state = stsrl::scenarios::jaw_worm(7);
    std::ranges::copy(cards, state.cards.hand.begin());
    state.cards.cardsInHand = int(cards.size());
    for (auto* pile : {&state.cards.discardPile, &state.cards.exhaustPile}) {
        pile->clear();
        for (const auto& card : cards) pile->push_back(card);
    }
    return state;
}

std::uint64_t play(const sts::BattleContext& state, int slot, int target = 0) {
    return action_key(state, Action{Type::CARD, slot, target});
}
std::uint64_t pick(const sts::BattleContext& state, int index) {
    return action_key(state, Action{Type::SINGLE_CARD_SELECT, index});
}
std::uint64_t pick_many(const sts::BattleContext& state, int positions) {
    return action_key(state, Action{Type::MULTI_CARD_SELECT, positions});
}
}  // namespace

TEST(ActionKey, PlayingACardIsTheCardAndTarget) {
    const auto hand = with_hand({{STRIKE_RED}, {BASH}, {STRIKE_RED}, {BASH, true}});
    EXPECT_EQ(play(hand, 0), play(hand, 2));
    EXPECT_NE(play(hand, 0), play(hand, 1));
    EXPECT_NE(play(hand, 1), play(hand, 3));
    EXPECT_NE(play(hand, 0, 0), play(hand, 0, 1));
    EXPECT_EQ(play(hand, 0), play(with_hand({{BASH}, {STRIKE_RED}}), 1));
}

TEST(ActionKey, PickingFromAPileIsTheCard) {
    const auto pile = headbutt({{STRIKE_RED}, {BASH}, {STRIKE_RED}});
    EXPECT_EQ(pick(pile, 0), pick(pile, 2));
    EXPECT_NE(pick(pile, 0), pick(pile, 1));
    EXPECT_EQ(pick(pile, 1), pick(headbutt({{STRIKE_RED}, {STRIKE_RED}, {BASH}}), 2));
}

TEST(ActionKey, PickingSeveralIsTheSetOfCards) {
    auto hand = with_hand({{STRIKE_RED}, {BASH}, {STRIKE_RED}});
    hand.inputState = sts::InputState::CARD_SELECT;
    hand.cardSelectInfo.cardSelectTask = sts::CardSelectTask::EXHAUST_MANY;
    EXPECT_EQ(pick_many(hand, 0b011), pick_many(hand, 0b110));
    EXPECT_NE(pick_many(hand, 0b011), pick_many(hand, 0b101));
}

TEST(ActionKey, UsingAPotionIsThePotionAndTarget) {
    sts::BattleContext state;
    state.potionCapacity = 3;
    state.potions[0] = state.potions[2] = sts::Potion::FIRE_POTION;
    state.potions[1] = sts::Potion::BLOCK_POTION;
    const auto use = [&](int slot, int target) { return action_key(state, Action{Type::POTION, slot, target}); };
    EXPECT_EQ(use(0, 0), use(2, 0));
    EXPECT_NE(use(0, 0), use(1, 0));
    EXPECT_NE(use(0, 0), use(0, 1));
}

TEST(ObservationKey, IgnoresTheOrderOfHandDiscardAndExhaust) {
    const auto state = with_piles({{STRIKE_RED}, {BASH}, {DEFEND_RED}});
    EXPECT_EQ(observation_key(state), observation_key(with_piles({{DEFEND_RED}, {STRIKE_RED}, {BASH}})));
    EXPECT_NE(observation_key(state), observation_key(with_piles({{DEFEND_RED}, {STRIKE_RED}, {BASH, true}})));
}
