#include "scenarios/slime_entry_projection.hpp"

#include "combat/BattleContext.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace stsrl::scenarios {
namespace {

SlimeEntryProjection parse_accepted(const nlohmann::json& row) {
    if (row.at("act") != 1 || row.at("boss").at("name") != "SLIME_BOSS"
        || row.at("encounter").at("name") != "SLIME_BOSS") {
        throw std::invalid_argument{"accepted entry is not a natural Act-1 Slime Boss root"};
    }
    SlimeEntryProjection result;
    result.source_seed = row.at("seed").get<std::uint64_t>();
    result.public_snapshot_sha256 = row.at("public_snapshot_sha256").get<std::string>();
    result.deck_signature = row.at("deck_signature").get<std::string>();
    result.entry_id = "slime-entry-v1-" + std::to_string(result.source_seed) + "-"
        + result.public_snapshot_sha256.substr(0, 16);
    result.hp = row.at("hp").get<int>();
    result.max_hp = row.at("max_hp").get<int>();
    if (result.hp <= 0 || result.max_hp < result.hp) throw std::invalid_argument{"invalid projected HP"};
    for (const auto& card : row.at("deck")) {
        const int upgrades = card.at("upgraded").get<int>();
        const int id = card.at("id").get<int>();
        if (upgrades < 0 || (upgrades > 1 && id != static_cast<int>(sts::CardId::SEARING_BLOW))) {
            throw std::invalid_argument{"only Searing Blow supports multi-upgrade projection"};
        }
        result.card_ids.push_back(id);
        result.upgrades.push_back(upgrades);
        result.misc.push_back(card.at("misc").get<int>());
    }
    return result;
}

}  // namespace

std::vector<SlimeEntryProjection> load_slime_entry_projections(
    const std::string& jsonl_path, int& skipped_nonaccepted) {
    std::ifstream stream{jsonl_path};
    if (!stream) throw std::runtime_error{"cannot open slime entry JSONL"};
    std::vector<SlimeEntryProjection> result;
    skipped_nonaccepted = 0;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        const auto row = nlohmann::json::parse(line);
        if (row.value("status", "") != "accepted") { ++skipped_nonaccepted; continue; }
        result.push_back(parse_accepted(row));
    }
    return result;
}

CombatEnvironment slime_entry_projection(const SlimeEntryProjection& entry, const std::uint64_t combat_seed) {
    if (entry.card_ids.size() != entry.upgrades.size() || entry.card_ids.size() != entry.misc.size()) {
        throw std::invalid_argument{"projected deck fields have inconsistent lengths"};
    }
    sts::GameContext game{sts::CharacterClass::IRONCLAD, combat_seed ? combat_seed : entry.source_seed, 1};
    game.floorNum = 16;
    game.curRoom = sts::Room::BOSS;
    game.curHp = entry.hp;
    game.maxHp = entry.max_hp;
    game.deck.cards.clear();
    game.deck.cardTypeCounts = {};
    game.deck.bottleIdxs = {-1, -1, -1};
    game.deck.upgradeableCount = game.deck.transformableCount = 0;
    game.relics.relics.clear();
    game.relics.relicBits0 = game.relics.relicBits1 = game.relics.relicBits2 = 0;
    game.potionCapacity = 0;
    game.potions = {};
    for (std::size_t i = 0; i < entry.card_ids.size(); ++i) {
        const auto id = static_cast<sts::CardId>(entry.card_ids[i]);
        sts::Card card{id, entry.upgrades[i] > 0};
        // lightspeed stores Searing Blow's upgrade count in misc; its external
        // entry representation carries the count in `upgraded`.
        card.misc = id == sts::CardId::SEARING_BLOW ? entry.upgrades[i] : entry.misc[i];
        game.deck.obtainRaw(card);
    }
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::SLIME_BOSS);
    return CombatEnvironment{std::move(battle)};
}

}  // namespace stsrl::scenarios
