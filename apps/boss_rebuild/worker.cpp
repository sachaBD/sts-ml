// Rebuild one recorded fight from its macro state and play it with guided-rollout MCTS.
//
//   boss_rebuild_worker REQUEST.json   -> one JSON result line on stdout
//
// request: {seed, ascension, act, floor, encounter, hp_before, simulations, state}
//   state: overworld_v1 macro state before the fight (deck with upgrades/misc, relics with data, potions,
//          potion_capacity, hp, max_hp, gold).
// The game is a fresh GameContext(seed) whose deck, relics, potions, HP, max HP and gold are replaced by the
// recorded ones (relics added directly: no on-pickup effects), set to the given act/floor in a boss room.
// Battle start uses Random(seed + floor) as in a real run, so with the original run seed and floor the
// combat streams (shuffle, AI, monster HP) start as in the original fight. miscRng / potionRng differ.
// Known gaps: which card a bottled relic holds is not recorded (no card is bottled), relic counters are
// whatever the macro state recorded. After init the player's HP is set to hp_before (battle-start HP of the
// original fight, i.e. after start-of-combat heals such as Pantograph), if given.
#include "agents/combat/search/teacher_search.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "combat/BattleContext.h"
#include "game/GameContext.h"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

sts::MonsterEncounter encounter_from(const std::string& name) {
    const int n = static_cast<int>(std::size(sts::monsterEncounterEnumNames));
    for (int i = 0; i < n; ++i)
        if (lower(sts::monsterEncounterEnumNames[i]) == name) return static_cast<sts::MonsterEncounter>(i);
    throw std::invalid_argument{"unknown encounter " + name};
}

sts::GameContext rebuild(const Json& req) {
    const auto& s = req.at("state");
    sts::GameContext gc{sts::CharacterClass::IRONCLAD, req.at("seed").get<std::uint64_t>(), req.at("ascension").get<int>()};
    gc.act = req.at("act").get<int>();
    gc.floorNum = req.at("floor").get<int>();
    gc.curRoom = sts::Room::BOSS;
    gc.curMapNodeX = -1;
    gc.curMapNodeY = 15;  // boss row: never the burning elite
    gc.maxHp = s.at("max_hp").get<int>();
    gc.curHp = s.at("hp").get<int>();
    gc.gold = s.at("gold").get<int>();

    gc.deck = sts::Deck{};
    for (const auto& c : s.at("deck")) {
        sts::Card card{static_cast<sts::CardId>(c.at("card_id").get<int>()), c.at("upgraded").get<int>()};
        card.misc = static_cast<std::int16_t>(c.at("misc").get<int>());
        gc.deck.obtainRaw(card);
    }
    gc.relics = sts::RelicContainer{};
    for (const auto& r : s.at("relics"))
        gc.relics.add({static_cast<sts::RelicId>(r.at("relic_id").get<int>()), r.at("data").get<int>()});

    gc.potionCapacity = s.at("potion_capacity").get<int>();
    std::fill(gc.potions.begin(), gc.potions.end(), sts::Potion::EMPTY_POTION_SLOT);
    gc.potionCount = 0;
    for (const auto& p : s.at("potions")) gc.potions.at(gc.potionCount++) = static_cast<sts::Potion>(p.at("potion_id").get<int>());

    const auto encounter = encounter_from(req.at("encounter").get<std::string>());
    gc.boss = encounter;
    gc.enterBattle(encounter);
    return gc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: boss_rebuild_worker REQUEST.json\n"; return 2; }
    std::ifstream in{argv[1]};
    const Json req = Json::parse(in);
    const auto gc = rebuild(req);
    sts::BattleContext battle;
    battle.init(gc);
    const int init_hp = battle.player.curHp;
    if (req.contains("hp_before") && !req.at("hp_before").is_null()) battle.player.curHp = req.at("hp_before").get<int>();
    const int start_hp = battle.player.curHp;
    const int start_potions = battle.potionCount;

    const auto sims = req.at("simulations").get<std::int64_t>();
    const auto search = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, stsrl::teacher::particles});
    const auto t0 = std::chrono::steady_clock::now();
    const auto end = stsrl::teacher::play_fight(battle, search, false, stsrl::teacher::particles, false);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    int monster_hp = 0, monster_max_hp = 0;
    for (int i = 0; i < end.monsters.monsterCount; ++i) {
        const auto& m = end.monsters.arr[i];
        if (m.isAlive()) monster_hp += m.curHp;
    }
    for (int i = 0; i < battle.monsters.monsterCount; ++i) monster_max_hp += battle.monsters.arr[i].maxHp;
    const Json out{{"won", end.outcome == sts::Outcome::PLAYER_VICTORY},
                   {"final_hp", end.player.curHp}, {"start_hp", start_hp}, {"init_hp", init_hp},
                   {"max_hp", end.player.maxHp}, {"turns", end.turn},
                   {"potions_start", start_potions}, {"potions_end", end.potionCount},
                   {"monster_hp_left", monster_hp}, {"monster_max_hp_start", monster_max_hp},
                   {"seconds", seconds}};
    std::cout << out.dump() << std::endl;
    return 0;
}
