#pragma once
#include "combat/BattleContext.h"
#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>

namespace stsrl::pv {
// Full future-state contents at callback-free boundaries. Dynamic containers are serialized by contents;
// GCC clears padding in trivial nested structs. Inactive hand/queue/limbo slots are omitted, not future state.
// Debug counters and other inactive inline fields are retained: distinct counts remain conservative upper bounds.
inline std::string turn_state_key(const sts::BattleContext& state, bool canonical = false,
                                  std::vector<std::pair<std::size_t, std::string>>* fields = nullptr) {
    if (state.actionQueue.size != 0)
        throw std::runtime_error{"turns: endpoint retains action callbacks; complete equality unavailable"};
    std::string key;
    auto append = [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        static_assert(std::is_trivially_copyable_v<T>);
        T copy = value;
        __builtin_clear_padding(&copy);
        key.append(reinterpret_cast<const char*>(&copy), sizeof(copy));
    };
#define M(field) do { if (fields) fields->emplace_back(key.size(), #field); append(state.field); } while (0)
    M(haveUsedDiscoveryAction); M(undefinedBehaviorEvoked); M(seed); M(floorNum); M(encounter);
    M(loopCount); M(energyWasted); M(cardsDrawn);
    M(aiRng); M(cardRandomRng); M(miscRng); M(monsterHpRng); M(potionRng); M(shuffleRng);
    M(ascension); M(outcome); M(unsupportedEffectKind); M(unsupportedEffectId); M(inputState); M(cardSelectInfo);
    M(monsterTurnIdx); M(isBattleOver); M(escapedCombat); M(endTurnQueued); M(endTurnAfterCurrentCard);
    M(turnHasEnded); M(skipMonsterTurn);
    if (state.cardQueue.size != 0) M(cardQueue); // Empty queue storage/indices have no future effect.
    M(nightmareCards); M(nightmareCopyCounts); M(potionCount); M(potionCapacity); M(potions); M(turn);
    M(player.cc); M(player.gold); M(player.curHp); M(player.maxHp);
    M(player.energy); M(player.energyPerTurn); M(player.cardDrawPerTurn); M(player.stance);
    M(player.orbSlots); M(player.orbs); M(player.orbData); M(player.lightningOrbsChanneled);
    M(player.frostOrbsChanneled); M(player.lastTargetedMonster); M(player.block); M(player.artifact);
    M(player.dexterity); M(player.focus); M(player.strength); M(player.justAppliedBits);
    M(player.statusBits0); M(player.statusBits1); M(player.relicBits0);
    M(player.relicBits1); M(player.happyFlowerCounter); M(player.incenseBurnerCounter); M(player.inkBottleCounter);
    M(player.inserterCounter); M(player.nunchakuCounter); M(player.penNibCounter); M(player.sundialCounter);
    M(player.haveUsedNecronomiconThisTurn); M(player.combustHpLoss); M(player.devaFormEnergyPerTurn); M(player.echoFormCardsDoubled);
    M(player.panacheCounter); M(player.cardsPlayedThisTurn); M(player.attacksPlayedThisTurn); M(player.skillsPlayedThisTurn);
    M(player.orangePelletsCardTypesPlayed); M(player.cardsDiscardedThisTurn); M(player.lastAttackUnblockedDamage); M(player.timesDamagedThisCombat);
    M(player.bomb1); M(player.bomb2); M(player.bomb3); M(player.selfRepairHeal);
    M(player.creativeAi); M(player.helloWorld); M(player.heatsinks); M(player.nirvanaBlock);
    M(player.rushdownDraw); M(player.studyInsights); M(player.mentalFortressBlock); M(player.machineLearning);
    M(player.storm);
    append(state.player.statusMap.size());
    for (const auto& [status, amount] : state.player.statusMap) { append(status); append(amount); }
    M(monsters); M(cards.nextUniqueCardId); M(cards.cardsInHand); M(cards.stasisCards);
    M(cards.handNormalityCount); M(cards.handPainCount); M(cards.strikeCount);
    M(cards.handBloodCardCount); M(cards.drawPileBloodCardCount); M(cards.discardPileBloodCardCount);
    if (state.cardQueue.size != 0 || state.inputState != sts::InputState::PLAYER_NORMAL) {
        M(curCardQueueItem); M(cards.limbo);
    }
    M(miscBits);
#undef M
    auto card_key = [&](const sts::CardInstance& card) {
        auto copy = card;
        __builtin_clear_padding(&copy);
        return std::string(reinterpret_cast<const char*>(&copy), sizeof(copy));
    };
    // Same content/count/sorted-pile pattern as state-fusion SearchState.cpp, but with full cards including uniqueId.
    auto append_cards = [&](const auto& pile, std::size_t count, bool sorted) {
        std::vector<std::string> cards;
        for (std::size_t i = 0; i < count; ++i) cards.push_back(card_key(pile[i]));
        if (sorted) std::sort(cards.begin(), cards.end());
        append(count);
        for (const auto& card : cards) key += card;
    };
    append_cards(state.cards.hand, state.cards.cardsInHand, canonical);
    append_cards(state.cards.drawPile, state.cards.drawPile.size(), false);
    append_cards(state.cards.discardPile, state.cards.discardPile.size(), canonical);
    append_cards(state.cards.exhaustPile, state.cards.exhaustPile.size(), canonical);
    return key;
}

} // namespace stsrl::pv
