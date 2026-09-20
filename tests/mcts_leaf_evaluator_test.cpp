#include "agents/mcts_agent.hpp"
#include "combat/environment.hpp"
#include "scenarios/slime_boss.hpp"

#include "combat/BattleContext.h"
#include "constants/Cards.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

void check(const bool condition) {
    if (!condition) {
        std::abort();
    }
}

sts::BattleContext lethal_jaw_worm_state() {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, 17, 1};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;

    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    battle.cards.cardsInHand = 1;
    battle.cards.hand[0] = sts::CardInstance{sts::CardId::STRIKE_RED};
    battle.monsters.arr[0].curHp = 1;
    battle.monsters.arr[0].block = 0;
    return battle;
}

}  // namespace

int main() {
    // A normal expansion must call the learned evaluator exactly once.
    auto normal_environment = stsrl::scenarios::slime_boss(1001);
    int normal_calls = 0;
    stsrl::MctsAgent normal_agent(
        7,
        {.simulations = 1},
        [&normal_calls](stsrl::CombatEnvironment& leaf) {
            check(!leaf.done());
            ++normal_calls;
            return 0.25;
        });
    const auto normal_result = normal_agent.search(normal_environment);
    check(normal_result.root_visits == 1);
    check(normal_calls == 1);

    // This root has one lethal Strike and one or more nonterminal actions. With
    // one simulation per root edge, every edge is expanded exactly once. The
    // lethal edge must receive the exact terminal return without invoking the
    // learned evaluator; every other edge must invoke it.
    stsrl::CombatEnvironment probe{lethal_jaw_worm_state()};
    const auto actions = probe.search_actions();
    check(actions.size() >= 2);

    std::size_t terminal_actions = 0;
    std::size_t terminal_action_index = 0;
    double terminal_value = 0.0;
    for (const auto& action : actions) {
        stsrl::CombatEnvironment successor{lethal_jaw_worm_state()};
        const auto successor_actions = successor.search_actions();
        const auto matching_action = std::find_if(
            successor_actions.begin(), successor_actions.end(),
            [&action](const auto& candidate) {
                return candidate.key == action.key;
            });
        check(matching_action != successor_actions.end());
        successor.step(matching_action->index);
        if (successor.done()) {
            ++terminal_actions;
            terminal_action_index = action.index;
            terminal_value = successor.combat_value();
        }
    }
    check(terminal_actions == 1);

    int learned_calls = 0;
    constexpr double learned_value = -0.25;
    stsrl::CombatEnvironment lethal_environment{lethal_jaw_worm_state()};
    stsrl::MctsAgent lethal_agent(
        11,
        {.simulations = actions.size()},
        [&learned_calls](stsrl::CombatEnvironment& leaf) {
            check(!leaf.done());
            ++learned_calls;
            return learned_value;
        });
    const auto result = lethal_agent.search(lethal_environment);

    check(result.root_visits == actions.size());
    check(learned_calls == static_cast<int>(actions.size() - terminal_actions));
    const auto terminal_stats = std::find_if(
        result.actions.begin(), result.actions.end(),
        [terminal_action_index](const auto& stats) {
            return stats.execution_index == terminal_action_index;
        });
    check(terminal_stats != result.actions.end());
    check(terminal_stats->visits == 1);
    check(std::abs(terminal_stats->q - terminal_value) < 1e-9);
    check(std::abs(terminal_stats->q - learned_value) > 1e-9);
}
