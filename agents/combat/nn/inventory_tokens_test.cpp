#include "agents/combat/nn/inventory_tokens.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

using namespace stsrl::nn;
namespace {
int kind(const stsrl::infra::Tokens& tokens, int row) { return int(tokens.data[row * width]); }
float feature(const stsrl::infra::Tokens& tokens, int row, int column) { return tokens.data[row * width + column]; }
}

TEST(InventoryTokens, InventoryConservesPileCountsAndDistinctModifications) {
    auto state = stsrl::scenarios::jaw_worm(1).battle();
    sts::CardInstance strike{sts::CardId::STRIKE_RED};
    auto cheap = strike;
    cheap.costForTurn = 0;
    auto modified = strike;
    modified.specialData = 3;
    state.cards.drawPile = {strike, strike, cheap, modified};
    state.cards.discardPile = {strike};
    state.cards.exhaustPile = {cheap};
    const auto tokens = encode_inventory(state, stsrl::combat::legal_actions(state));
    std::array<float, 3> counts{};
    int variants = 0;
    for (int i = 0; i < tokens.count; ++i) if (kind(tokens, i) == int(InventoryKind::inventory)) {
        ++variants;
        for (int pile = 0; pile < 3; ++pile) counts[pile] += feature(tokens, i, 3 + 16 + pile) * 10;
    }
    EXPECT_EQ(variants, 3);
    EXPECT_FLOAT_EQ(counts[0], state.cards.drawPile.size());
    EXPECT_FLOAT_EQ(counts[1], state.cards.discardPile.size());
    EXPECT_FLOAT_EQ(counts[2], state.cards.exhaustPile.size());
}

TEST(InventoryTokens, UnknownPileOrderDoesNotAffectInputs) {
    auto state = stsrl::scenarios::jaw_worm(2).battle();
    const auto actions = stsrl::combat::legal_actions(state);
    const auto expected = encode_inventory(state, actions);
    std::reverse(state.cards.drawPile.begin(), state.cards.drawPile.end());
    EXPECT_EQ(encode_inventory(state, actions).data, expected.data);
}

TEST(InventoryTokens, ActionReferencesFollowSourcesAndTargetInstances) {
    auto state = stsrl::scenarios::jaw_worm(3).battle();
    state.monsters.arr[1] = state.monsters.arr[0];
    state.monsters.arr[1].idx = 1;
    state.cards.cardsInHand = 1;
    state.cards.hand[0] = sts::CardInstance{sts::CardId::TRIP};
    state.obtainPotion(sts::Potion::FIRE_POTION);
    using Action = sts::search::Action;
    using Type = sts::search::ActionType;
    const std::vector<Action> actions{{Type::CARD, 0, 0}, {Type::CARD, 0, 1},
                                      {Type::POTION, 0, 0}, {Type::POTION, 0, 1}, {Type::END_TURN}};
    for (int order = 0; order < 2; ++order) {
        const auto tokens = encode_inventory(state, actions);
        const int first = tokens.count - tokens.actions;
        for (int i = 0; i < 4; ++i) {
            const int target = int(feature(tokens, first + i, inventory_target)) - 1;
            const int source = int(feature(tokens, first + i, inventory_source)) - 1;
            ASSERT_GE(target, 0);
            ASSERT_GE(source, 0);
            EXPECT_EQ(kind(tokens, target), int(InventoryKind::monster));
            EXPECT_EQ(kind(tokens, source), int(i < 2 ? InventoryKind::card : InventoryKind::potion));
            EXPECT_FLOAT_EQ(feature(tokens, target, 4), state.monsters.arr[i % 2].curHp / 100.f);
        }
        EXPECT_NE(feature(tokens, first, inventory_target), feature(tokens, first + 1, inventory_target));
        EXPECT_EQ(feature(tokens, tokens.count - 1, inventory_target), 0);
        state.monsters.arr[0].curHp /= 2;
        std::swap(state.monsters.arr[0], state.monsters.arr[1]);
        state.monsters.arr[0].idx = 0;
        state.monsters.arr[1].idx = 1;
    }
}

TEST(InventoryTokens, EffectsHaveOwnersAndAllEntitiesAttend) {
    auto state = stsrl::scenarios::jaw_worm(4).battle();
    state.player.strength = -2;
    state.player.setHasStatus<PS::BARRICADE>(true);
    state.monsters.arr[0].strength = -3;
    const auto actions = stsrl::combat::legal_actions(state);
    const auto tokens = encode(state, actions, false, Encoding::inventory_v1);
    bool player_strength = false, monster_strength = false;
    for (int i = 0; i < tokens.count; ++i) {
        const auto type = static_cast<InventoryKind>(kind(tokens, i));
        if (type == InventoryKind::player_effect || type == InventoryKind::monster_effect) {
            const int owner = int(feature(tokens, i, inventory_owner)) - 1;
            ASSERT_GE(owner, 0);
            EXPECT_EQ(kind(tokens, owner), int(type == InventoryKind::player_effect ? InventoryKind::player : InventoryKind::monster));
            if (type == InventoryKind::player_effect && feature(tokens, i, 1) == int(PS::STRENGTH)) {
                player_strength = true;
                EXPECT_LT(feature(tokens, i, 3), 0);
            }
            if (type == InventoryKind::monster_effect && feature(tokens, i, 1) == int(sts::MS::STRENGTH) + 1) {
                monster_strength = true;
                EXPECT_LT(feature(tokens, i, 3), 0);
            }
        }
        EXPECT_EQ(type == InventoryKind::action, i >= tokens.count - tokens.actions);
        for (int column : {inventory_owner, inventory_source, inventory_target}) {
            const float reference = feature(tokens, i, column);
            EXPECT_GE(reference, 0);
            EXPECT_LE(reference, tokens.count);
            EXPECT_EQ(reference, std::floor(reference));
        }
    }
    EXPECT_TRUE(player_strength);
    EXPECT_TRUE(monster_strength);
    EXPECT_EQ(tokens.attended, tokens.count);
    EXPECT_EQ(tokens.data[2], inventory_version);
}
