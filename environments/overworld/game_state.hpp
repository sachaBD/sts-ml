// The persistent game state that carries between fights (combat_transition_v1 `pre` / `post`), as JSON.
// Shared by apps/combat_transition and apps/gauntlet (model inputs for the pre-combat outcome model).
#pragma once
#include "constants/Cards.h"
#include "constants/Potions.h"
#include "constants/Relics.h"
#include "game/GameContext.h"

#include <cctype>
#include <string>

#include <nlohmann/json.hpp>

namespace stsrl::game_state {
using Json = nlohmann::json;

inline std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline Json card(const sts::Card& c) {
    return {{"card_id", static_cast<int>(c.id)}, {"upgraded", c.getUpgraded()}, {"misc", c.misc},
            {"name", lower(sts::cardEnumStrings[static_cast<int>(c.id)])},
            {"mechanics", {{"type", static_cast<int>(c.getType())}, {"rarity", static_cast<int>(c.getRarity())},
                           {"cost", sts::getEnergyCost(c.id, c.upgraded)}, {"base_damage", c.getBaseDamage()},
                           {"innate", c.isInnate()}, {"ethereal", sts::isCardEthereal(c.id, c.upgraded)},
                           {"exhaust", sts::doesCardExhaust(c.id, c.upgraded)},
                           {"self_retain", sts::doesCardSelfRetain(c.id, c.upgraded)}, {"x_cost", sts::isXCost(c.id)}}}};
}

inline Json state(const sts::GameContext& g) {
    Json deck = Json::array(), relics = Json::array(), potions = Json::array();
    for (const auto& c : g.deck.cards) deck.push_back(card(c));
    for (const auto& r : g.relics.relics)
        relics.push_back({{"relic_id", static_cast<int>(r.id)}, {"data", r.data},
                          {"name", lower(sts::relicEnumNames[static_cast<int>(r.id)])}});
    for (int i = 0; i < g.potionCapacity; ++i) {
        const auto p = g.potions[static_cast<std::size_t>(i)];
        if (p == sts::Potion::EMPTY_POTION_SLOT || p == sts::Potion::INVALID) continue;
        potions.push_back({{"potion_id", static_cast<int>(p)}, {"name", lower(sts::potionEnumNames[static_cast<int>(p)])}});
    }
    return {{"hp", g.curHp}, {"max_hp", g.maxHp}, {"gold", g.gold}, {"floor", g.floorNum},
            {"potion_capacity", g.potionCapacity}, {"deck", deck}, {"relics", relics}, {"potions", potions}};
}

}  // namespace stsrl::game_state
