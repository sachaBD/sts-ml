#include "agents/mcts_agent.hpp"
#include "combat/environment.hpp"
#include "scenarios/jaw_worm.hpp"
#include "scenarios/slime_boss.hpp"
#include "combat/BattleContext.h"
#include "constants/Cards.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include "sim/search/BattleScumSearcher2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace {
void check(bool value) { if (!value) std::abort(); }

sts::BattleContext state() {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, 7, 1};
    game.floorNum = 1; game.curRoom = sts::Room::MONSTER;
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    battle.cards.cardsInHand = 3;
    battle.cards.hand[0] = sts::CardInstance{sts::CardId::STRIKE_RED};
    battle.cards.hand[1] = sts::CardInstance{sts::CardId::BASH};
    battle.cards.hand[2] = sts::CardInstance{sts::CardId::DEFEND_RED};
    battle.player.strength = 2;
    battle.player.dexterity = 2;
    battle.player.setStatusValueNoChecks<PlayerStatus::WEAK>(1);
    battle.player.setHasStatus<PlayerStatus::WEAK>(true);
    battle.player.setStatusValueNoChecks<PlayerStatus::FRAIL>(1);
    battle.player.setHasStatus<PlayerStatus::FRAIL>(true);
    battle.monsters.arr[0].addDebuff<sts::MonsterStatus::VULNERABLE>(1);
    return battle;
}

void compare_execution(sts::CardId id, bool damage) {
    auto battle = state();
    stsrl::CombatEnvironment environment{battle};
    const auto encoding = environment.decision().encoding;
    sts::search::BattleScumSearcher2 searcher{battle};
    sts::search::BattleScumSearcher2::Node node;
    searcher.enumerateActionsForNode(node, battle);
    for (const auto& edge : node.edges) {
        const auto source = edge.action.getSourceIdx();
        if (edge.action.getActionType() != sts::search::ActionType::CARD || battle.cards.hand[source].id != id) continue;
        if (damage) {
            const auto interaction = *std::find_if(encoding.card_monster_interactions.begin(), encoding.card_monster_interactions.end(),
                [&](const auto& x) { return encoding.cards[x.card_index].card_id == static_cast<int>(id); });
            const int before = battle.monsters.arr[edge.action.getTargetIdx()].curHp;
            edge.action.execute(battle);
            check(before - battle.monsters.arr[edge.action.getTargetIdx()].curHp == int(interaction.numeric[2] * 100));
        } else {
            const auto card = *std::find_if(encoding.cards.begin(), encoding.cards.end(), [&](const auto& x) { return x.card_id == static_cast<int>(id) && x.zone == stsrl::CardZone::hand; });
            const int before = battle.player.block;
            edge.action.execute(battle);
            check(battle.player.block - before == int(card.numeric[6] * 50));
        }
        return;
    }
    std::abort();
}
}

int main() {
    auto original = stsrl::scenarios::jaw_worm(1234);
    const auto expected = original.decision().encoding;
    check(expected.version == stsrl::combat_encoding_schema_version);
    check(expected == original.determinized(1).decision().encoding);
    check(expected == original.determinized(999).decision().encoding);

    const auto decision = original.decision();
    check(decision.encoding.legal_actions.size() == decision.legal_actions.size());
    std::set<std::size_t> indices;
    for (const auto& action : decision.encoding.legal_actions) { check(action.execution_index < decision.legal_actions.size()); indices.insert(action.execution_index); }
    check(indices.size() == decision.legal_actions.size());
    compare_execution(sts::CardId::STRIKE_RED, true);
    compare_execution(sts::CardId::BASH, true);
    compare_execution(sts::CardId::DEFEND_RED, false);
    auto first_state = state();
    auto reordered_state = first_state;
    std::swap(reordered_state.cards.hand[0], reordered_state.cards.hand[1]);
    stsrl::CombatEnvironment first_environment{std::move(first_state)};
    stsrl::CombatEnvironment reordered_environment{std::move(reordered_state)};
    const auto first_actions = first_environment.search_actions();
    const auto reordered_actions = reordered_environment.search_actions();
    check(first_actions.size() == reordered_actions.size());
    for (const auto& action : first_actions)
        check(std::any_of(reordered_actions.begin(), reordered_actions.end(), [&](const auto& other) { return other.key == action.key; }));
    const auto strike = *std::find_if(first_actions.begin(), first_actions.end(), [](const auto& x) { return x.key.card_id == static_cast<int>(sts::CardId::STRIKE_RED); });
    const auto bash = *std::find_if(first_actions.begin(), first_actions.end(), [](const auto& x) { return x.key.card_id == static_cast<int>(sts::CardId::BASH); });
    check(strike.key != bash.key);
    auto target_a = strike.key;
    auto target_b = target_a;
    ++target_b.target_vulnerable;
    check(target_a != target_b);
    target_b = target_a;
    ++target_b.target_move;
    check(target_a != target_b);
    auto duplicate_state = state();
    duplicate_state.cards.cardsInHand = 2;
    duplicate_state.cards.hand[1] = duplicate_state.cards.hand[0];
    stsrl::CombatEnvironment duplicate_environment{std::move(duplicate_state)};
    const auto duplicate_actions = duplicate_environment.search_actions();
    check(std::count_if(duplicate_actions.begin(), duplicate_actions.end(), [](const auto& x) { return x.key.card_id == static_cast<int>(sts::CardId::STRIKE_RED); }) == 1);
    auto mcts_environment = stsrl::scenarios::slime_boss(1);
    stsrl::MctsAgent mcts{1, {.simulations = 16, .rollout_limit = 32}};
    const auto result = mcts.search(mcts_environment);
    check(result.root_visits == 16 && std::isfinite(result.root_value) && result.root_value >= -1 && result.root_value <= 1);
    check(std::any_of(result.actions.begin(), result.actions.end(), [&](const auto& x) { return x.execution_index == result.chosen_action && std::isfinite(x.q) && x.q >= -1 && x.q <= 1; }));
    check(!mcts_environment.action_description(result.chosen_action).empty());
}
