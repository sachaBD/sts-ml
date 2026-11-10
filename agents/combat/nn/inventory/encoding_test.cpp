#include "agents/combat/nn/inventory/encoding.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>

using namespace stsrl::nn;
namespace {
const auto encoder = make_encoder(load_layout(STSRL_SOURCE_DIR "/agents/combat/nn/inventory/layout.toml"));
const auto& layout = encoder->layout();
stsrl::infra::Tokens encode_inventory(const sts::BattleContext& state, std::span<const sts::search::Action> actions) {
    return encoder->encode(state, actions);
}
int kind(const stsrl::infra::Tokens& tokens, int row) { return int(tokens.data[row * layout.width]); }
float feature(const stsrl::infra::Tokens& tokens, int row, int column) { return tokens.data[row * layout.width + column]; }
const int source = layout.link("source"), target_column = layout.link("target");
}

TEST(InventoryEncoding, ModifiedCardsInTheDrawPileStayVisible) {
    auto state = stsrl::scenarios::jaw_worm(1);
    const sts::CardInstance strike{sts::CardId::STRIKE_RED};
    state.cards.drawPile = {strike, strike};
    const auto actions = stsrl::combat::legal_actions(state);
    const auto expected = encode_inventory(state, actions);
    state.cards.drawPile[0].costForTurn = 0;
    EXPECT_NE(encode_inventory(state, actions).data, expected.data);
}

TEST(InventoryEncoding, HiddenDrawOrderDoesNotLeakIntoInputs) {
    auto state = stsrl::scenarios::jaw_worm(2);
    const auto actions = stsrl::combat::legal_actions(state);
    const auto expected = encode_inventory(state, actions);
    std::reverse(state.cards.drawPile.begin(), state.cards.drawPile.end());
    EXPECT_EQ(encode_inventory(state, actions).data, expected.data);
}

TEST(InventoryEncoding, ActionsPointAtTheirSourceCardAndTargetMonster) {
    auto state = stsrl::scenarios::jaw_worm(3);
    state.monsters.arr[1] = state.monsters.arr[0];
    state.monsters.arr[1].idx = 1;
    state.monsters.monsterCount = 2;
    state.monsters.monstersAlive = 2;
    state.cards.cardsInHand = 2;
    state.cards.hand[0] = state.cards.hand[1] = sts::CardInstance{sts::CardId::TRIP};
    state.obtainPotion(sts::Potion::FIRE_POTION);
    state.obtainPotion(sts::Potion::FIRE_POTION);
    using Action = sts::search::Action;
    using Type = sts::search::ActionType;
    const std::vector<Action> actions{{Type::CARD, 0, 0}, {Type::CARD, 1, 1},
                                      {Type::POTION, 0, 0}, {Type::POTION, 1, 1}, {Type::END_TURN}};
    for (int order = 0; order < 2; ++order) {
        const auto tokens = encode_inventory(state, actions);
        const int first = tokens.count - tokens.actions;
        for (int i = 0; i < 4; ++i) {
            const int target = int(feature(tokens, first + i, target_column)) - layout.link_offset;
            const int from = int(feature(tokens, first + i, source)) - layout.link_offset;
            ASSERT_GE(target, 0);
            ASSERT_GE(from, 0);
            EXPECT_EQ(kind(tokens, target), layout.kind("monster").id);
            EXPECT_EQ(kind(tokens, from), layout.kind(i < 2 ? "card" : "potion").id);
            EXPECT_FLOAT_EQ(feature(tokens, target, 4), state.monsters.arr[i % 2].curHp / 100.f);
        }
        EXPECT_NE(feature(tokens, first, source), feature(tokens, first + 1, source));
        EXPECT_NE(feature(tokens, first + 2, source), feature(tokens, first + 3, source));
        EXPECT_NE(feature(tokens, first, target_column), feature(tokens, first + 1, target_column));
        EXPECT_EQ(feature(tokens, tokens.count - 1, target_column), 0);
        state.monsters.arr[0].curHp /= 2;
        std::swap(state.monsters.arr[0], state.monsters.arr[1]);
        state.monsters.arr[0].idx = 0;
        state.monsters.arr[1].idx = 1;
    }
}
