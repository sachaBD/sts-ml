#include "environments/combat/record_v4.hpp"

#include "constants/CharacterClasses.h"
#include "constants/Rooms.h"
#include "sim/search/Action.h"

#include <stdexcept>
#include <string>

namespace stsrl::combat_v4 {
namespace {

using Json = nlohmann::json;

Json rng_json(const sts::Random& r) { return {{"counter", r.counter}, {"seed0", r.seed0}, {"seed1", r.seed1}}; }

void rng_restore(sts::Random& r, const Json& j) {
    r.counter = j.at("counter").get<std::int32_t>();
    r.seed0 = j.at("seed0").get<std::uint64_t>();
    r.seed1 = j.at("seed1").get<std::uint64_t>();
}

bool at_burning_elite(const sts::GameContext& gc) {
    return gc.map && gc.map->burningEliteX == gc.curMapNodeX && gc.map->burningEliteY == gc.curMapNodeY;
}

}  // namespace

Json start_json(const sts::GameContext& gc) {
    if (gc.cc != sts::CharacterClass::IRONCLAD) throw std::invalid_argument{"combat_v4: Ironclad only"};
    if (gc.potionCapacity < 0 || gc.potionCapacity > 5) throw std::invalid_argument{"combat_v4: potion capacity"};
    Json potions = Json::array(), relics = Json::array(), deck = Json::array(), bottled = Json::array();
    for (int i = 0; i < gc.potionCapacity; ++i) potions.push_back(static_cast<int>(gc.potions[i]));
    for (int i = gc.potionCapacity; i < 5; ++i)
        if (gc.potions[i] != sts::Potion::EMPTY_POTION_SLOT && gc.potions[i] != sts::Potion::INVALID)
            throw std::invalid_argument{"combat_v4: potion beyond capacity"};
    for (const auto& r : gc.relics.relics) relics.push_back({{"id", static_cast<int>(r.id)}, {"data", r.data}});
    for (const auto& c : gc.deck.cards)
        deck.push_back({{"id", static_cast<int>(c.id)}, {"upgraded", c.upgraded}, {"misc", c.misc}});
    for (int i = 0; i < 3; ++i) bottled.push_back(gc.deck.bottleIdxs[i]);
    return {{"seed", gc.seed}, {"ascension", gc.ascension}, {"act", gc.act}, {"floor", gc.floorNum},
            {"encounter", static_cast<int>(gc.info.encounter)}, {"cur_room", static_cast<int>(gc.curRoom)},
            {"last_room", static_cast<int>(gc.lastRoom)},
            {"burning_elite_buff", at_burning_elite(gc) ? gc.map->burningEliteBuff : -1},
            {"hp", gc.curHp}, {"max_hp", gc.maxHp}, {"gold", gc.gold},
            {"misc_rng", rng_json(gc.miscRng)}, {"potion_rng", rng_json(gc.potionRng)},
            {"potion_capacity", gc.potionCapacity}, {"potions", potions}, {"relics", relics}, {"deck", deck},
            {"bottled", bottled}};
}

Start start_from_json(const Json& s) {
    const auto rng = [](const Json& j) {
        return RngState{j.at("counter").get<std::int32_t>(), j.at("seed0").get<std::uint64_t>(),
                        j.at("seed1").get<std::uint64_t>()};
    };
    Start r{};
    r.seed = s.at("seed").get<std::uint64_t>();
    r.ascension = s.at("ascension"); r.act = s.at("act"); r.floor = s.at("floor");
    r.encounter = s.at("encounter"); r.cur_room = s.at("cur_room"); r.last_room = s.at("last_room");
    r.burning_elite_buff = s.at("burning_elite_buff"); r.hp = s.at("hp"); r.max_hp = s.at("max_hp");
    r.gold = s.at("gold");
    r.misc_rng = rng(s.at("misc_rng")); r.potion_rng = rng(s.at("potion_rng"));
    r.potion_capacity = s.at("potion_capacity");
    for (const auto& p : s.at("potions")) r.potions.push_back(p.get<int>());
    for (const auto& x : s.at("relics")) r.relics.push_back({x.at("id").get<int>(), x.at("data").get<int>()});
    for (const auto& c : s.at("deck"))
        r.deck.push_back({c.at("id").get<int>(), c.at("upgraded").get<bool>(), c.at("misc").get<int>()});
    const auto& bottled = s.at("bottled");
    if (bottled.size() != 3) throw std::invalid_argument{"combat_v4: bottled"};
    for (int i = 0; i < 3; ++i) r.bottled[i] = bottled[i].get<int>();
    return r;
}

sts::GameContext start_game(const Json& s) { return start_game(start_from_json(s)); }

sts::GameContext start_game(const Start& s) {
    sts::GameContext gc{sts::CharacterClass::IRONCLAD, s.seed, s.ascension};
    gc.act = s.act;
    gc.floorNum = s.floor;
    gc.info.encounter = static_cast<sts::MonsterEncounter>(s.encounter);
    gc.curRoom = static_cast<sts::Room>(s.cur_room);
    gc.lastRoom = static_cast<sts::Room>(s.last_room);
    if (s.burning_elite_buff >= 0) {
        gc.curMapNodeX = gc.map->burningEliteX;
        gc.curMapNodeY = gc.map->burningEliteY;
        gc.map->burningEliteBuff = s.burning_elite_buff;
    } else {
        gc.curMapNodeX = -7;  // no map node: never the burning elite
        gc.curMapNodeY = -7;
    }
    gc.curHp = s.hp;
    gc.maxHp = s.max_hp;
    gc.gold = s.gold;
    gc.miscRng.counter = s.misc_rng.counter; gc.miscRng.seed0 = s.misc_rng.seed0; gc.miscRng.seed1 = s.misc_rng.seed1;
    gc.potionRng.counter = s.potion_rng.counter; gc.potionRng.seed0 = s.potion_rng.seed0;
    gc.potionRng.seed1 = s.potion_rng.seed1;
    gc.potionCapacity = s.potion_capacity;
    std::fill(gc.potions.begin(), gc.potions.end(), sts::Potion::EMPTY_POTION_SLOT);
    gc.potionCount = 0;
    if (static_cast<int>(s.potions.size()) != gc.potionCapacity) throw std::invalid_argument{"combat_v4: potion slots"};
    for (std::size_t i = 0; i < s.potions.size(); ++i) {
        gc.potions[i] = static_cast<sts::Potion>(s.potions[i]);
        if (gc.potions[i] != sts::Potion::EMPTY_POTION_SLOT) ++gc.potionCount;
    }
    gc.relics = sts::RelicContainer{};
    for (const auto& r : s.relics) gc.relics.add({static_cast<sts::RelicId>(r.id), r.data});
    gc.deck = sts::Deck{};
    for (const auto& c : s.deck) {
        sts::Card card{static_cast<sts::CardId>(c.id)};
        card.upgraded = c.upgraded;
        card.misc = static_cast<std::int16_t>(c.misc);
        gc.deck.obtainRaw(card);
    }
    for (int i = 0; i < 3; ++i) gc.deck.bottleIdxs[i] = s.bottled[i];
    gc.screenState = sts::ScreenState::BATTLE;
    return gc;
}

sts::BattleContext replay(const Json& start, const std::vector<std::uint32_t>& actions) {
    const auto gc = start_game(start);
    sts::BattleContext bc;
    bc.init(gc);
    for (std::size_t i = 0; i < actions.size(); ++i) {
        if (bc.outcome != sts::Outcome::UNDECIDED)
            throw std::runtime_error{"combat_v4 replay: fight ended before action " + std::to_string(i)};
        const sts::search::Action action{actions[i]};
        if (!action.isValidAction(bc))
            throw std::runtime_error{"combat_v4 replay: invalid action " + std::to_string(i)};
        action.execute(bc);
    }
    if (bc.outcome == sts::Outcome::UNDECIDED) {
        if (bc.turn >= 50 || actions.size() >= 512) {
            bc.outcome = sts::Outcome::PLAYER_LOSS;
            bc.player.curHp = 0;
        } else {
            throw std::runtime_error{"combat_v4 replay: fight did not end"};
        }
    }
    return bc;
}

}  // namespace stsrl::combat_v4
