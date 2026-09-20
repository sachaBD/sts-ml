#include "scenarios/jaw_worm.hpp"

#include "combat/BattleContext.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"

#include <utility>

namespace stsrl::scenarios {

CombatEnvironment jaw_worm(const std::uint64_t seed) {
    constexpr auto ascension = 1;

    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, ascension};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;

    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    return CombatEnvironment{std::move(battle)};
}

}  // namespace stsrl::scenarios
