#include "scenarios/slime_boss.hpp"

#include "combat/BattleContext.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"

#include <stdexcept>
#include <utility>

namespace stsrl::scenarios {
namespace {

sts::BattleContext make_slime_boss(const std::uint64_t seed) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, 1};
    game.floorNum = 16;
    game.curRoom = sts::Room::BOSS;
    game.curHp = game.maxHp = 80;
    for (int i = 0; i < game.deck.size(); ++i)
        if (game.deck.cards[i].id == sts::CardId::BASH) { game.deck.upgrade(i); break; }
    game.deck.obtain(game, sts::CardId::FLAME_BARRIER);
    game.deck.obtain(game, sts::CardId::COMBUST);
    game.deck.obtain(game, sts::CardId::HEMOKINESIS);
    game.deck.upgrade(game.deck.size() - 1);
    game.deck.obtain(game, sts::CardId::BATTLE_TRANCE);
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::SLIME_BOSS);
    return battle;
}

void validate_hp(const sts::BattleContext& battle, const int hp) {
    const auto maximum = battle.monsters.arr.front().maxHp;
    if (hp <= 0 || hp > maximum) throw std::invalid_argument{"Slime Boss HP must be positive and no greater than its maximum HP"};
}

} // namespace

CombatEnvironment slime_boss(const std::uint64_t seed) {
    return CombatEnvironment{make_slime_boss(seed)};
}

CombatEnvironment slime_boss_after_split(const std::uint64_t seed, const int hp) {
    auto battle = make_slime_boss(seed);
    validate_hp(battle, hp);
    if (hp > battle.monsters.arr.front().maxHp / 2)
        throw std::invalid_argument{"Slime Boss split HP must not exceed half its maximum HP"};
    sts::Monster::slimeBossSplit(battle, hp);
    return CombatEnvironment{std::move(battle)};
}

} // namespace stsrl::scenarios
