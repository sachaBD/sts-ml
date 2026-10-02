// Public/history-reconstructible A20 Ironclad overworld observations. No RNG state or future lists.
#pragma once
#include "environments/overworld/game_state.hpp"
#include "constants/RelicPools.h"
#include "game/Shop.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace stsrl::overworld {
using Json = nlohmann::json;

inline Json rarity_probabilities(int factor, int rare, int uncommon) {
    std::array<int, 3> count{}; // common, uncommon, rare, for the NEXT individual roll
    for (int roll = 0; roll < 100; ++roll)
        ++count[roll + factor < rare ? 2 : roll + factor < rare + uncommon ? 1 : 0];
    return Json{{"common", count[0] / 100.0}, {"uncommon", count[1] / 100.0}, {"rare", count[2] / 100.0}};
}

inline Json question_probabilities(const sts::GameContext& gc, bool previous_shop) {
    std::array<int, 4> count{}; // fight, shop, treasure, event
    const bool forced = gc.hasRelic(sts::RelicId::TINY_CHEST) && gc.relics.getRelicValue(sts::RelicId::TINY_CHEST) == 3;
    const int monster = static_cast<int>(gc.monsterChance * 100);
    const int shop = monster + (previous_shop ? 0 : static_cast<int>(gc.shopChance * 100));
    const int treasure = shop + static_cast<int>(gc.treasureChance * 100);
    for (int roll = 0; roll < 100; ++roll) {
        int idx = forced ? 2 : roll < monster ? 0 : roll < shop ? 1 : roll < treasure ? 2 : 3;
        if (idx == 0 && gc.hasRelic(sts::RelicId::JUZU_BRACELET)) idx = 3;
        ++count[idx];
    }
    return {{"fight", count[0] / 100.0}, {"shop", count[1] / 100.0},
            {"treasure", count[2] / 100.0}, {"event", count[3] / 100.0}};
}

inline Json public_encounters(const sts::GameContext& gc) {
    using ME = sts::MonsterEncounter;
    Json elite = Json::array(), hallway = Json::array(), history = Json::array();
    // Only consumed prefix entries are accessed: already-entered room encounters are public.
    const int m = std::min(gc.monsterListOffset, static_cast<int>(gc.monsterList.size()));
    const int e = std::min(gc.eliteMonsterListOffset, static_cast<int>(gc.eliteMonsterList.size()));
    const ME last = m ? gc.monsterList[m - 1] : ME::INVALID;
    const ME prev = m > 1 ? gc.monsterList[m - 2] : ME::INVALID;
    const ME last_elite = e ? gc.eliteMonsterList[e - 1] : ME::INVALID;
    for (int i = 0; i < m; ++i) history.push_back(static_cast<int>(gc.monsterList[i]));
    if (gc.act >= 1 && gc.act <= 3) {
        for (auto id : sts::MonsterEncounterPool::elites[gc.act - 1])
            if (id != last_elite) elite.push_back(static_cast<int>(id));
        const int weak_limit = gc.act == 1 ? 3 : 2;
        const bool weak = m < weak_limit;
        const bool first_strong = m == weak_limit;
        const auto* ids = weak ? sts::MonsterEncounterPool::weakEnemies[gc.act - 1]
                               : sts::MonsterEncounterPool::strongEnemies[gc.act - 1];
        const auto* weights = weak ? sts::MonsterEncounterPool::weakWeights[gc.act - 1]
                                   : sts::MonsterEncounterPool::strongWeights[gc.act - 1];
        const int size = weak ? sts::MonsterEncounterPool::weakCount[gc.act - 1]
                              : sts::MonsterEncounterPool::strongCount[gc.act - 1];
        double total = 0;
        for (int i = 0; i < size; ++i) {
            const auto id = ids[i];
            bool allowed = id != last && id != prev;
            // First strong encounter has special exclusions, not the ordinary last-two rejection.
            if (first_strong) {
                allowed = !((last == ME::SMALL_SLIMES && (id == ME::LARGE_SLIME || id == ME::LOTS_OF_SLIMES)) ||
                            (last == ME::TWO_LOUSE && id == ME::THREE_LOUSE));
            }
            if (allowed) { hallway.push_back({{"id", static_cast<int>(id)}, {"probability", weights[i]}}); total += weights[i]; }
        }
        for (auto& item : hallway) item["probability"] = item["probability"].get<double>() / total;
    } else if (gc.act == 4 && !e) {
        elite.push_back(static_cast<int>(ME::SHIELD_AND_SPEAR));
    }
    return {{"possible_elites", elite}, {"last_elite", static_cast<int>(last_elite)},
            {"hallway_history", history}, {"hallway_count", m}, {"elite_count", e},
            {"next_hallway", hallway}, {"history_scope", "current_act_consumed_prefix"}};
}

inline Json observation_json(const sts::GameContext& gc) {
    using R = sts::RelicId;
    Json events = Json::object();
    auto event_pool = [&](const auto& pool, int kind) {
        Json out = Json::array();
        for (auto id : pool) {
            const bool eligible = kind == 0 ? gc.canAddEvent(id) : kind == 2 ? gc.canAddOneTimeEvent(id)
                : !(id == sts::Event::MATCH_AND_KEEP && sts::GameContext::disableMatchAndKeep);
            out.push_back({{"id", static_cast<int>(id)}, {"eligible", eligible}});
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a["id"] < b["id"]; });
        return out;
    };
    events["normal"] = event_pool(gc.eventList, 0);
    events["shrine"] = event_pool(gc.shrineList, 1);
    events["one_time"] = event_pool(gc.specialOneTimeEventList, 2);
    Json relics = Json::object();
    auto relic_pool = [&](const auto& pool) {
        Json out = Json::array();
        for (auto id : pool) if (!gc.hasRelic(id))
            out.push_back({{"id", static_cast<int>(id)}, {"eligible", gc.relicCanSpawn(id, false)},
                           {"shop_eligible", gc.relicCanSpawn(id, true)}});
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a["id"] < b["id"]; });
        return out;
    };
    // Do NOT expose actual shuffled pool membership: invisible spawn rejections consume relics.
    // Public candidates are an upper bound, not a claim of exact remaining inventory.
    if (gc.cc == sts::CharacterClass::IRONCLAD) {
        relics["common"] = relic_pool(sts::Ironclad::commonRelicPool);
        relics["uncommon"] = relic_pool(sts::Ironclad::uncommonRelicPool);
        relics["rare"] = relic_pool(sts::Ironclad::rareRelicPool);
        relics["shop"] = relic_pool(sts::Ironclad::shopRelicPool);
        relics["boss"] = relic_pool(sts::Ironclad::bossRelicPool);
    }
    const int gift = gc.hasRelic(R::NLOTHS_GIFT) ? 3 : 1;
    const double potion_roll = std::clamp(gc.hasRelic(R::WHITE_BEAST_STATUE) ? 100 : 40 + gc.potionChance, 0, 100) / 100.0;
    return {{"version", 1}, {"character", static_cast<int>(gc.cc)}, {"ascension", gc.ascension},
            {"act", gc.act}, {"floor_in_act", gc.curMapNodeY + 1},
            {"keys", {{"ruby", gc.redKey}, {"sapphire", gc.blueKey}, {"emerald", gc.greenKey}}},
            {"card_rarity_factor", gc.cardRarityFactor},
            {"card_rarity", {{"hallway", rarity_probabilities(gc.cardRarityFactor, 3 * gift, 37)},
                             {"elite", rarity_probabilities(gc.cardRarityFactor, 10 * gift, 40)},
                             {"shop", rarity_probabilities(gc.cardRarityFactor, 9, 37)}}},
            {"potion_chance_modifier", gc.potionChance}, {"potion_roll_probability", potion_roll},
            {"potion_obtain_probability", gc.hasRelic(R::SOZU) ? 0.0 : potion_roll},
            {"question_base", {{"fight", gc.monsterChance}, {"shop", gc.shopChance}, {"treasure", gc.treasureChance}}},
            {"question_next", question_probabilities(gc, gc.curRoom == sts::Room::SHOP)},
            {"question_after_shop", question_probabilities(gc, true)},
            {"current_room", static_cast<int>(gc.curRoom)},
            {"visible_encounter", gc.screenState == sts::ScreenState::BATTLE ? static_cast<int>(gc.info.encounter) : 0},
            {"shop_remove_count", gc.shopRemoveCount},
            {"shop_remove_cost", sts::Shop::getRemoveCost(gc)}, {"events", events},
            {"relic_candidates", relics}, {"relic_candidates_exact", false},
            {"encounters", public_encounters(gc)}};
}
} // namespace stsrl::overworld
