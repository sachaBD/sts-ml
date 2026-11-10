#include "environments/combat/fight_start.hpp"
#include "constants/CharacterClasses.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include <algorithm>
#include <stdexcept>

namespace stsrl::combat {
namespace {
using Json = nlohmann::json;

void restore(sts::Random& r, const Json& j) {
    r.counter = j.at("counter").get<std::int32_t>();
    r.seed0 = j.at("seed0").get<std::uint64_t>();
    r.seed1 = j.at("seed1").get<std::uint64_t>();
}

sts::GameContext game(const Json& s) {
    sts::GameContext gc{sts::CharacterClass::IRONCLAD, s.at("seed").get<std::uint64_t>(), s.at("ascension").get<int>()};
    gc.act = s.at("act");
    gc.floorNum = s.at("floor");
    gc.info.encounter = static_cast<sts::MonsterEncounter>(s.at("encounter").get<int>());
    gc.curRoom = static_cast<sts::Room>(s.at("cur_room").get<int>());
    gc.lastRoom = static_cast<sts::Room>(s.at("last_room").get<int>());
    if (const int buff = s.at("burning_elite_buff"); buff >= 0) {
        gc.curMapNodeX = gc.map->burningEliteX;
        gc.curMapNodeY = gc.map->burningEliteY;
        gc.map->burningEliteBuff = buff;
    } else {
        gc.curMapNodeX = -7;  // no map node: never the burning elite
        gc.curMapNodeY = -7;
    }
    gc.curHp = s.at("hp");
    gc.maxHp = s.at("max_hp");
    gc.gold = s.at("gold");
    restore(gc.miscRng, s.at("misc_rng"));
    restore(gc.potionRng, s.at("potion_rng"));
    gc.potionCapacity = s.at("potion_capacity");
    std::fill(gc.potions.begin(), gc.potions.end(), sts::Potion::EMPTY_POTION_SLOT);
    gc.potionCount = 0;
    const auto& potions = s.at("potions");
    if (static_cast<int>(potions.size()) != gc.potionCapacity) throw std::invalid_argument{"fight start: potion slots"};
    for (std::size_t i = 0; i < potions.size(); ++i) {
        gc.potions[i] = static_cast<sts::Potion>(potions[i].get<int>());
        if (gc.potions[i] != sts::Potion::EMPTY_POTION_SLOT) ++gc.potionCount;
    }
    gc.relics = sts::RelicContainer{};
    for (const auto& r : s.at("relics")) gc.relics.add({static_cast<sts::RelicId>(r.at("id").get<int>()), r.at("data").get<int>()});
    gc.deck = sts::Deck{};
    for (const auto& c : s.at("deck")) {
        sts::Card card{static_cast<sts::CardId>(c.at("id").get<int>())};
        card.upgraded = c.at("upgraded").get<bool>();
        card.misc = static_cast<std::int16_t>(c.at("misc").get<int>());
        gc.deck.obtainRaw(card);
    }
    const auto& bottled = s.at("bottled");
    if (bottled.size() != 3) throw std::invalid_argument{"fight start: bottled"};
    for (int i = 0; i < 3; ++i) gc.deck.bottleIdxs[i] = bottled[i].get<int>();
    gc.screenState = sts::ScreenState::BATTLE;
    return gc;
}
}  // namespace

sts::BattleContext start_battle(const Json& start) {
    sts::BattleContext bc;
    bc.init(game(start));
    return bc;
}
}  // namespace stsrl::combat
