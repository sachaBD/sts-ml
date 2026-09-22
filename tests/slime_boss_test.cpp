#include "scenarios/slime_boss.hpp"
#include "constants/MonsterIds.h"
#include "constants/Cards.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>

#ifdef NDEBUG
#undef assert
#define assert(expression) ((expression) ? static_cast<void>(0) : std::abort())
#endif

int main() {
    const auto initial = stsrl::scenarios::slime_boss(42).decision().encoding;
    assert(initial.version == 3);
    assert(initial.global.numeric.size() == 50);
    const auto card_count = [&](sts::CardId id) { return std::count_if(initial.cards.begin(), initial.cards.end(), [&](const auto& card) { return card.card_id == static_cast<int>(id); }); };
    assert(card_count(sts::CardId::FLAME_BARRIER) == 1 && card_count(sts::CardId::COMBUST) == 1);
    assert(card_count(sts::CardId::HEMOKINESIS) == 1 && card_count(sts::CardId::BATTLE_TRANCE) == 1);
    assert(std::any_of(initial.cards.begin(), initial.cards.end(), [](const auto& card) { return card.card_id == static_cast<int>(sts::CardId::BATTLE_TRANCE) && card.numeric[7] == .3f; }));
    assert(std::any_of(initial.cards.begin(), initial.cards.end(), [](const auto& card) { return card.card_id == static_cast<int>(sts::CardId::BASH) && card.numeric[0] == 1; }));
    assert(initial.monsters.size() == 1);
    assert(initial.monsters.front().monster_id == static_cast<int>(sts::MonsterId::SLIME_BOSS));

    constexpr int split_hp = 17;
    const auto split = stsrl::scenarios::slime_boss_after_split(42, split_hp).decision().encoding;
    assert(split.monsters.size() == 2);
    assert(std::count_if(split.monsters.begin(), split.monsters.end(), [](const auto& monster) {
        return monster.monster_id == static_cast<int>(sts::MonsterId::SPIKE_SLIME_L) && monster.numeric[0] * 100 == split_hp;
    }) == 1);
    assert(std::count_if(split.monsters.begin(), split.monsters.end(), [](const auto& monster) {
        return monster.monster_id == static_cast<int>(sts::MonsterId::ACID_SLIME_L) && monster.numeric[0] * 100 == split_hp;
    }) == 1);
}
