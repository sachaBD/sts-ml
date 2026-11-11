// Features must tell the network the truth about the game, and cover every card a supported deck can hold.
#include "agents/combat/nn/features.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include "constants/CardPools.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <optional>

using namespace stsrl::nn;

namespace {
// Strength, dexterity, weak, frail and vulnerable all change the real numbers the features must report.
sts::BattleContext modified_fight() {
    auto battle = stsrl::scenarios::jaw_worm(7);
    battle.cards.cardsInHand = 3;
    battle.cards.hand[0] = sts::CardInstance{sts::CardId::STRIKE_RED};
    battle.cards.hand[1] = sts::CardInstance{sts::CardId::BASH};
    battle.cards.hand[2] = sts::CardInstance{sts::CardId::DEFEND_RED};
    battle.player.strength = 2;
    battle.player.dexterity = 2;
    battle.player.setStatusValueNoChecks<PlayerStatus::WEAK>(1);
    battle.player.setHasStatus<PlayerStatus::WEAK>(true);
    battle.player.setStatusValueNoChecks<PlayerStatus::FRAIL>(1);
    battle.player.setHasStatus<PlayerStatus::FRAIL>(true);
    battle.monsters.arr[0].addDebuff<sts::MonsterStatus::VULNERABLE>(1);
    return battle;
}
}  // namespace

TEST(Features, DamageAndBlockMatchWhatTheSimulatorDoes) {
    const auto start = modified_fight();
    int checked = 0;
    for (const auto action : stsrl::combat::legal_actions(start)) {
        if (action.getActionType() != sts::search::ActionType::CARD) continue;
        const auto id = start.cards.hand[action.getSourceIdx()].id;
        const auto features = encode_action(start, action);
        auto after = start;
        action.execute(after);
        if (id == sts::CardId::DEFEND_RED) {
            EXPECT_EQ(after.player.block - start.player.block, int(features.source_card->numeric[6] * 50));
        } else {
            const int target = action.getTargetIdx();
            EXPECT_EQ(start.monsters.arr[target].curHp - after.monsters.arr[target].curHp,
                      int((*features.interaction)[2] * 100)) << sts::getCardEnumName(id);
        }
        ++checked;
    }
    EXPECT_EQ(checked, 3);
}

TEST(Features, EveryIroncladAndColorlessCardEncodes) {
    std::vector<sts::CardId> cards(sts::baseColorlessPool.begin(), sts::baseColorlessPool.end());
    for (int id = 1; id <= int(sts::CardId::ZAP); ++id)  // ZAP is the last CardId
        if (sts::getCardColor(sts::CardId(id)) == sts::CardColor::RED) cards.push_back(sts::CardId(id));
    for (const auto id : cards) {
        auto battle = modified_fight();
        battle.cards.cardsInHand = 1;
        battle.cards.hand[0] = sts::CardInstance{id};
        EXPECT_NO_THROW(EXPECT_EQ(encode_cards(battle).size(), 1u + battle.cards.drawPile.size())) << sts::getCardEnumName(id);
    }
}

namespace {
// Every pile holds different cards, so a card's id says where it came from.
sts::BattleContext selecting(sts::CardSelectTask task) {
    using enum sts::CardId;
    auto state = stsrl::scenarios::jaw_worm(7);
    state.inputState = sts::InputState::CARD_SELECT;
    state.cardSelectInfo.cardSelectTask = task;
    state.cardSelectInfo.cards = {CLEAVE, FLEX, ANGER};
    state.cardSelectInfo.data0 = 1;
    const auto fill = [](auto& pile, std::vector<sts::CardInstance> cards) {
        pile.clear();
        for (const auto& card : cards) pile.push_back(card);
    };
    fill(state.cards.drawPile, {{SHRUG_IT_OFF}, {TRUE_GRIT}});
    fill(state.cards.discardPile, {{BASH}, {CLOTHESLINE}});
    fill(state.cards.exhaustPile, {{INFLAME}, {CARNAGE}});
    state.cards.hand[0] = {IRON_WAVE};
    state.cards.hand[1] = {TWIN_STRIKE};
    state.cards.cardsInHand = 2;
    return state;
}

std::array<std::vector<int>, 4> piles(const sts::BattleContext& state) {
    const auto ids = [](const auto& pile) {
        std::vector<int> out;
        for (const auto& card : pile) out.push_back(int(card.id));
        std::ranges::sort(out);
        return out;
    };
    const auto& c = state.cards;
    return {ids(std::span{c.hand.data(), std::size_t(c.cardsInHand)}), ids(c.drawPile), ids(c.discardPile),
            ids(c.exhaustPile)};
}

// The card a selection takes, found by playing it: the one that leaves its pile, or for an offer, the one that appears.
std::optional<std::pair<int, CardZone>> taken(const sts::BattleContext& state, sts::search::Action action) {
    auto after = state;
    action.execute(after);
    const auto before = piles(state), now = piles(after);
    constexpr std::array zones{CardZone::hand, CardZone::draw, CardZone::discard, CardZone::exhaust};
    for (std::size_t z = 0; z < zones.size(); ++z) {
        std::vector<int> left;
        std::ranges::set_difference(before[z], now[z], std::back_inserter(left));
        if (!left.empty()) return std::pair{left.front(), zones[z]};
    }
    for (std::size_t z = 0; z < zones.size(); ++z) {
        std::vector<int> appeared;
        std::ranges::set_difference(now[z], before[z], std::back_inserter(appeared));
        if (!appeared.empty()) return std::pair{appeared.front(), CardZone::offered};
    }
    return std::nullopt;
}

class SelectionTokens : public testing::TestWithParam<sts::CardSelectTask> {};
}  // namespace

INSTANTIATE_TEST_SUITE_P(EveryPile, SelectionTokens,
                         testing::Values(sts::CardSelectTask::WARCRY, sts::CardSelectTask::HEADBUTT,
                                         sts::CardSelectTask::EXHUME, sts::CardSelectTask::SEEK,
                                         sts::CardSelectTask::DISCOVERY, sts::CardSelectTask::CODEX),
                         [](const auto& info) { return std::string{sts::cardSelectTaskStrings[int(info.param)]}; });

TEST_P(SelectionTokens, ShowTheCardTheMoveTakesAndWhereFrom) {
    const auto state = selecting(GetParam());
    for (const auto action : stsrl::combat::legal_actions(state)) {
        const auto card = encode_action(state, action).source_card;
        const auto expected = taken(state, action);
        ASSERT_EQ(card.has_value(), expected.has_value());
        if (card) { EXPECT_EQ(std::pair(card->card_id, card->zone), *expected); }
    }
}
