#include "environments/combat/scenarios/jaw_worm.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"

namespace stsrl::scenarios {
sts::BattleContext jaw_worm(const std::uint64_t seed) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, 1};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    return battle;
}
}  // namespace stsrl::scenarios
