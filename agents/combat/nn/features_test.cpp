// Features must tell the network the truth about the game, and cover every card a supported deck can hold.
#include "agents/combat/nn/features.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include "constants/CardPools.h"
#include <gtest/gtest.h>
#include <algorithm>

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
