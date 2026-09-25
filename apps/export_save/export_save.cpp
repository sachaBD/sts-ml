// Prototype: one stored combat_v3 fight -> an STS save file (IRONCLAD.autosave) that resumes at the start of
// that fight in the real game client. Best effort; unsupported cases throw.
//   export_save REQUEST.json OUT.autosave
//   REQUEST.json: the fight replay request (apps/common/replay.py): {run_seed, ascension, fight_index, actions}.
// The act 1 run is replayed up to the fight (apps/common/fight_replay.hpp); the game right before it is written
// as the save the client makes on entering the fight's room (SaveFile.SaveType.ENTER_ROOM). The client accepts
// plain JSON saves (SaveFileObfuscator.isObfuscated: no '{'), so the file is not obfuscated.
// Check: the save is loaded back with the simulator's own loader (GameContext::initFromSave) and the resulting
// fight start must print the same as the replayed one.
#include "apps/common/fight_replay.hpp"
#include "game/SaveFile.h"
#include "sim/PrintHelpers.h"

#include <nlohmann/json.hpp>
// The Java names (to_json of the ids). The header defines a non-inline from_json, already in sts_lightspeed's
// SaveFile.cpp: rename its from_json functions here (only to_json is used).
#define from_json export_save_unused_from_json
#include "constants/SaveFileMappings.h"
#undef from_json

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using Json = nlohmann::json;
using sts::RelicId;

// sts_lightspeed's relic data -> the Java relic counter (-1: no counter; -2: used up).
int relic_counter(const sts::RelicInstance& relic) {
    switch (relic.id) {
        case RelicId::GIRYA:
        case RelicId::HAPPY_FLOWER:
        case RelicId::INCENSE_BURNER:
        case RelicId::INK_BOTTLE:
        case RelicId::INSERTER:
        case RelicId::MATRYOSHKA:
        case RelicId::NUNCHAKU:
        case RelicId::OMAMORI:
        case RelicId::PEN_NIB:
        case RelicId::SUNDIAL:
        case RelicId::TINY_CHEST:
            return relic.data;
        case RelicId::LIZARD_TAIL:
        case RelicId::MAW_BANK:
            return relic.data ? -1 : -2;
        case RelicId::NEOWS_LAMENT:
            return relic.data > 0 ? relic.data : -2;
        case RelicId::NLOTHS_HUNGRY_FACE:
            throw std::runtime_error{"unsupported relic: N'loth's Hungry Face"};
        default:
            return -1;
    }
}

template <typename List>
Json names(const List& list, std::size_t from = 0) {
    Json out = Json::array();
    for (std::size_t i = from; i < static_cast<std::size_t>(list.size()); ++i) out.push_back(list[i]);
    return out;
}

// The save the client writes on entering the fight's room, from the game right before the fight.
Json save_json(const sts::GameContext& game) {
    if (game.act != 1) throw std::runtime_error{"only act 1 is supported"};
    std::string room;
    int room_x = game.curMapNodeX, room_y = game.curMapNodeY;
    Json monsters, elites;
    switch (game.curRoom) {
        case sts::Room::MONSTER:
        case sts::Room::ELITE: {
            // A "?" room that rolled a fight: the client would re-roll it from the event RNG before the roll.
            if (game.map->getNode(room_x, room_y).room != game.curRoom)
                throw std::runtime_error{"fights from ? rooms are not supported"};
            const bool elite = game.curRoom == sts::Room::ELITE;
            if ((elite ? game.eliteMonsterListOffset : game.monsterListOffset) < 1)
                throw std::runtime_error{"monster list offset is 0 at the fight"};
            room = elite ? "com.megacrit.cardcrawl.rooms.MonsterRoomElite" : "com.megacrit.cardcrawl.rooms.MonsterRoom";
            // The client removes a fight from its list on leaving the room: the current one is still the head.
            monsters = names(game.monsterList, game.monsterListOffset - (elite ? 0 : 1));
            elites = names(game.eliteMonsterList, game.eliteMonsterListOffset - (elite ? 1 : 0));
            break;
        }
        case sts::Room::BOSS:
            room = "com.megacrit.cardcrawl.rooms.MonsterRoomBoss";
            room_x = -1;
            room_y = 15;
            monsters = names(game.monsterList, game.monsterListOffset);
            elites = names(game.eliteMonsterList, game.eliteMonsterListOffset);
            break;
        default:  // e.g. an event's fight (Dead Adventurer, ...)
            throw std::runtime_error{"fights started by events are not supported"};
    }

    Json cards = Json::array();
    for (int i = 0; i < game.deck.size(); ++i) {
        const auto& card = game.deck.cards[i];
        const bool searing = card.id == sts::CardId::SEARING_BLOW;
        cards.push_back({{"id", card.id}, {"upgrades", searing ? card.misc : (card.upgraded ? 1 : 0)},
                         {"misc", searing ? 0 : card.misc}});
    }
    Json relics = Json::array(), counters = Json::array();
    for (const auto& relic : game.relics.relics) {
        relics.push_back(relic.id);
        counters.push_back(relic_counter(relic));
    }
    Json potions = Json::array();
    for (int i = 0; i < game.potionCapacity; ++i) potions.push_back(game.potions[i]);

    // The client re-runs the relics' onEnterRoom on load: undo what the simulator already applied.
    int gold = game.gold;
    if (game.relics.has(RelicId::MAW_BANK) && game.relics.getRelicValue(RelicId::MAW_BANK) != 0) gold -= 12;

    Json boss_list = Json::array({game.boss});
    if (game.secondBoss != sts::MonsterEncounter::INVALID) boss_list.push_back(game.secondBoss);

    Json save = {
        {"name", "sts_combat_rl"}, {"loadout", ""},
        {"current_health", game.curHp}, {"max_health", game.maxHp}, {"max_orbs", 0}, {"gold", gold},
        {"hand_size", 5}, {"potion_slots", game.potionCapacity}, {"red", 3}, {"green", 0}, {"blue", 0},
        {"cards", cards}, {"obtained_cards", Json::object()},
        {"relics", relics}, {"relic_counters", counters},
        {"blights", Json::array()}, {"blight_counters", Json::array()}, {"endless_increments", Json::array()},
        {"potions", potions},
        {"is_ascension_mode", game.ascension > 0}, {"ascension_level", game.ascension},
        {"chose_neow_reward", true}, {"level_name", "Exordium"},
        {"play_time", 0}, {"save_date", 0}, {"daily_date", 0},
        {"floor_num", game.floorNum}, {"act_num", game.act},
        {"seed", static_cast<std::int64_t>(game.seed)}, {"special_seed", 0}, {"seed_set", true},
        {"is_trial", false}, {"is_daily", false}, {"is_final_act_on", false}, {"is_endless_mode", false},
        {"has_ruby_key", game.redKey}, {"has_emerald_key", game.greenKey}, {"has_sapphire_key", game.blueKey},
        {"custom_mods", Json::array()}, {"daily_mods", Json::array()},
        {"monster_seed_count", game.monsterRng.counter}, {"event_seed_count", game.eventRng.counter},
        {"merchant_seed_count", game.merchantRng.counter}, {"card_seed_count", game.cardRng.counter},
        {"treasure_seed_count", game.treasureRng.counter}, {"relic_seed_count", game.relicRng.counter},
        {"potion_seed_count", game.potionRng.counter},
        // Reseeded from seed + floor on entering a room; stored for completeness only.
        {"monster_hp_seed_count", 0}, {"ai_seed_count", 0}, {"shuffle_seed_count", 0},
        {"card_random_seed_count", 0},
        {"card_random_seed_randomizer", game.cardRarityFactor}, {"potion_chance", game.potionChance},
        {"purgeCost", 75 + 25 * game.shopRemoveCount},
        {"monster_list", monsters}, {"elite_monster_list", elites}, {"boss_list", boss_list}, {"boss", game.boss},
        {"event_list", names(game.eventList)}, {"one_time_event_list", names(game.specialOneTimeEventList)},
        {"event_chances", {0.0f, game.monsterChance, game.shopChance, game.treasureChance}},
        // Only the current node: the client marks the visited path with these (cosmetic).
        {"path_x", {room_x}}, {"path_y", {room_y}}, {"room_x", room_x}, {"room_y", room_y},
        {"spirit_count", 0}, {"current_room", room},
        {"common_relics", names(game.commonRelicPool)}, {"uncommon_relics", names(game.uncommonRelicPool)},
        {"rare_relics", names(game.rareRelicPool)}, {"shop_relics", names(game.shopRelicPool)},
        {"boss_relics", names(game.bossRelicPool)},
        {"post_combat", false}, {"mugged", false}, {"smoked", false}, {"combat_rewards", Json::array()},
        {"monsters_killed", 0}, {"elites1_killed", 0}, {"elites2_killed", 0}, {"elites3_killed", 0},
        {"champions", 0}, {"perfect", 0}, {"overkill", false}, {"combo", false}, {"cheater", false},
        {"gold_gained", 0}, {"mystery_machine", 0},
        {"metric_campfire_rested", 0}, {"metric_campfire_upgraded", 0}, {"metric_campfire_rituals", 0},
        {"metric_campfire_meditates", 0}, {"metric_purchased_purges", 0},
        {"metric_build_version", "sts_combat_rl"}, {"metric_seed_played", std::to_string(game.seed)},
        {"metric_floor_reached", game.floorNum}, {"metric_playtime", 0},
        {"neow_bonus", nullptr}, {"neow_cost", nullptr},
    };
    for (const auto* key : {"metric_potions_floor_spawned", "metric_potions_floor_usage", "metric_current_hp_per_floor",
                            "metric_max_hp_per_floor", "metric_gold_per_floor", "metric_path_per_floor",
                            "metric_path_taken", "metric_items_purchased", "metric_item_purchase_floors",
                            "metric_items_purged", "metric_items_purged_floors", "metric_card_choices",
                            "metric_event_choices", "metric_boss_relics", "metric_damage_taken",
                            "metric_potions_obtained", "metric_relics_obtained", "metric_campfire_choices"})
        save[key] = Json::array();
    const std::array<const char*, 3> bottles{"bottled_flame", "bottled_lightning", "bottled_tornado"};
    for (int b = 0; b < 3; ++b) {
        const auto index = game.deck.bottleIdxs[b];
        if (index < 0) continue;
        const auto& card = cards[index];
        save[bottles[b]] = card["id"];
        save[std::string{bottles[b]} + "_upgrade"] = card["upgrades"];
        save[std::string{bottles[b]} + "_misc"] = card["misc"];
    }
    // An id without a Java name maps to null.
    for (const auto* key : {"cards", "relics", "potions", "monster_list", "elite_monster_list", "boss_list",
                            "event_list", "one_time_event_list", "common_relics", "uncommon_relics",
                            "rare_relics", "shop_relics", "boss_relics"})
        for (const auto& item : save[key])
            if (item.is_null() || (item.is_object() && item["id"].is_null()))
                throw std::runtime_error{std::string{"an id in "} + key + " has no Java name"};
    return save;
}

std::string printed(const sts::BattleContext& battle) {
    std::ostringstream out;
    out << battle;
    return out.str();
}

// Loads the save with the simulator's own loader: the fight it starts must be the replayed one.
void check(const Json& save, const sts::BattleContext& start) {
    sts::GameContext loaded;
    loaded.initFromSave(sts::SaveFile{save.dump(), sts::CharacterClass::IRONCLAD});
    sts::BattleContext battle;
    battle.init(loaded);
    if (printed(battle) != printed(start)) {
        std::cerr << "replayed fight start:\n" << printed(start) << "\nfrom the save:\n" << printed(battle) << '\n';
        throw std::runtime_error{"the save doesn't start the replayed fight"};
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: export_save REQUEST.json OUT.autosave\n";
        return 2;
    }
    try {
        std::ifstream file{argv[1]};
        if (!file) throw std::runtime_error{std::string{"cannot read "} + argv[1]};
        const auto request = Json::parse(file);
        Json save;
        stsrl::replay::to_fight(request, [&](const sts::GameContext& game, const sts::BattleContext& start, const Json&) {
            save = save_json(game);
            check(save, start);
            return start;  // the replayed run stops after this fight
        });
        std::ofstream out{argv[2]};
        if (!(out << save.dump(2) << '\n')) throw std::runtime_error{std::string{"cannot write "} + argv[2]};
        std::cerr << "wrote " << argv[2] << ": floor " << save["floor_num"] << ", " << save["current_room"] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "export_save: " << error.what() << '\n';
        return 1;
    }
}
