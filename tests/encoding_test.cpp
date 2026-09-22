#include "agents/mcts_agent.hpp"
#include "combat/environment.hpp"
#include "scenarios/jaw_worm.hpp"
#include "scenarios/slime_boss.hpp"
#include "combat/BattleContext.h"
#include "constants/Cards.h"
#include "constants/CardPools.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include "sim/search/BattleScumSearcher2.h"

#include <algorithm>
#include <array>
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

    auto multi_base = state();
    multi_base.monsters.monsterCount = 2;
    multi_base.monsters.monstersAlive = 2;
    multi_base.monsters.arr[1] = multi_base.monsters.arr[0];
    multi_base.monsters.arr[1].idx = 1;

    {
        auto identical_state = multi_base;
        stsrl::CombatEnvironment identical_env{std::move(identical_state)};
        const auto actions = identical_env.search_actions();
        check(std::count_if(actions.begin(), actions.end(), [](const auto& x) {
            return x.key.card_id == static_cast<int>(sts::CardId::STRIKE_RED);
        }) == 1);
    }

    {
        auto vuln_state = multi_base;
        vuln_state.monsters.arr[0].vulnerable = 0;
        vuln_state.monsters.arr[1].vulnerable = 1;
        stsrl::CombatEnvironment vuln_env{std::move(vuln_state)};
        const auto actions = vuln_env.search_actions();
        std::vector<stsrl::SearchActionKey> strike_keys;
        for (const auto& x : actions) {
            if (x.key.card_id == static_cast<int>(sts::CardId::STRIKE_RED)) {
                strike_keys.push_back(x.key);
            }
        }
        check(strike_keys.size() == 2);
        check(strike_keys[0] != strike_keys[1]);
        check(strike_keys[0].target_vulnerable != strike_keys[1].target_vulnerable);
    }

    {
        auto move_state = multi_base;
        move_state.monsters.arr[0].vulnerable = 0;
        move_state.monsters.arr[1].vulnerable = 0;
        move_state.monsters.arr[0].moveHistory[0] = sts::MMID::JAW_WORM_CHOMP;
        move_state.monsters.arr[1].moveHistory[0] = sts::MMID::JAW_WORM_BELLOW;
        stsrl::CombatEnvironment move_env{std::move(move_state)};
        const auto actions = move_env.search_actions();
        std::vector<stsrl::SearchActionKey> strike_keys;
        for (const auto& x : actions) {
            if (x.key.card_id == static_cast<int>(sts::CardId::STRIKE_RED)) {
                strike_keys.push_back(x.key);
            }
        }
        check(strike_keys.size() == 2);
        check(strike_keys[0] != strike_keys[1]);
        check(strike_keys[0].target_move != strike_keys[1].target_move);
    }

    {
        auto select_state = state();
        select_state.inputState = sts::InputState::CARD_SELECT;
        select_state.cardSelectInfo.cardSelectTask = sts::CardSelectTask::ARMAMENTS;
        select_state.cardSelectInfo.pickCount = 1;
        stsrl::CombatEnvironment select_env{select_state};
        stsrl::CombatEnvironment repeat_env{std::move(select_state)};
        const auto selected = select_env.search_actions();
        const auto repeated = repeat_env.search_actions();
        check(!selected.empty() && selected.size() == repeated.size());
        for (const auto& action : selected) check(std::any_of(repeated.begin(), repeated.end(), [&](const auto& other) { return other.key == action.key; }));
    }
    {
        auto select_state = state();
        select_state.inputState = sts::InputState::CARD_SELECT;
        select_state.cardSelectInfo.cardSelectTask = sts::CardSelectTask::EXHAUST_MANY;
        select_state.cardSelectInfo.pickCount = 2;
        stsrl::CombatEnvironment select_env{std::move(select_state)};
        const auto selected = select_env.search_actions();
        check(std::any_of(selected.begin(), selected.end(), [](const auto& action) {
            return action.key.kind == static_cast<int>(sts::search::ActionType::MULTI_CARD_SELECT);
        }));
    }

    // Every card observed across the accepted 23-entry pilot must encode at a
    // combat root; this is the minimum natural-deck admission contract.
    constexpr std::array pilot_cards{
        sts::CardId::ANGER, sts::CardId::ARMAMENTS, sts::CardId::BANDAGE_UP, sts::CardId::BARRICADE,
        sts::CardId::BASH, sts::CardId::BATTLE_TRANCE, sts::CardId::BLUDGEON, sts::CardId::BODY_SLAM,
        sts::CardId::BRUTALITY, sts::CardId::CARNAGE, sts::CardId::CLEAVE, sts::CardId::CLOTHESLINE,
        sts::CardId::COMBUST, sts::CardId::CORRUPTION, sts::CardId::DARK_EMBRACE, sts::CardId::DEEP_BREATH,
        sts::CardId::DEFEND_RED, sts::CardId::DEMON_FORM, sts::CardId::DISARM, sts::CardId::DROPKICK,
        sts::CardId::ENTRENCH, sts::CardId::EVOLVE, sts::CardId::EXHUME, sts::CardId::FEEL_NO_PAIN,
        sts::CardId::FIEND_FIRE, sts::CardId::FIRE_BREATHING, sts::CardId::FLAME_BARRIER, sts::CardId::FLEX,
        sts::CardId::GHOSTLY_ARMOR, sts::CardId::HEADBUTT, sts::CardId::HEAVY_BLADE, sts::CardId::IMMOLATE,
        sts::CardId::IMPERVIOUS, sts::CardId::INFERNAL_BLADE, sts::CardId::INFLAME, sts::CardId::IRON_WAVE,
        sts::CardId::MADNESS, sts::CardId::METALLICIZE, sts::CardId::OFFERING, sts::CardId::PANACEA,
        sts::CardId::POMMEL_STRIKE, sts::CardId::POWER_THROUGH, sts::CardId::PUMMEL, sts::CardId::RAGE,
        sts::CardId::RAMPAGE, sts::CardId::SECOND_WIND, sts::CardId::SHOCKWAVE, sts::CardId::SHRUG_IT_OFF,
        sts::CardId::SPOT_WEAKNESS, sts::CardId::STRIKE_RED, sts::CardId::SWORD_BOOMERANG, sts::CardId::THUNDERCLAP,
        sts::CardId::TRUE_GRIT, sts::CardId::UPPERCUT, sts::CardId::WHIRLWIND, sts::CardId::WILD_STRIKE};
    for (const auto id : pilot_cards) {
        auto pilot_state = state();
        pilot_state.cards.cardsInHand = 1;
        pilot_state.cards.hand[0] = sts::CardInstance{id};
        stsrl::CombatEnvironment pilot_environment{std::move(pilot_state)};
        const auto encoded = pilot_environment.decision().encoding;
        check(std::any_of(encoded.cards.begin(), encoded.cards.end(), [&](const auto& card) { return card.card_id == static_cast<int>(id); }));
    }
    for (const auto id : sts::baseColorlessPool) {
        auto colorless_state = state();
        colorless_state.cards.cardsInHand = 1;
        colorless_state.cards.hand[0] = sts::CardInstance{id};
        stsrl::CombatEnvironment colorless_environment{std::move(colorless_state)};
        check(!colorless_environment.decision().encoding.cards.empty());
    }
    for (const bool upgraded : {false, true}) {
        auto finesse_state = state();
        finesse_state.cards.cardsInHand = 1;
        finesse_state.cards.hand[0] = sts::CardInstance{sts::CardId::FINESSE, upgraded};
        stsrl::CombatEnvironment finesse_environment{std::move(finesse_state)};
        const auto finesse_encoding = finesse_environment.decision().encoding;
        const auto finesse = *std::find_if(finesse_encoding.cards.begin(), finesse_encoding.cards.end(), [](const auto& token) { return token.card_id == static_cast<int>(sts::CardId::FINESSE); });
        check(finesse.numeric[5] == (upgraded ? 4.f : 2.f) / 50.f);
        check(finesse.numeric[7] == .1f);
    }
    {
        auto random_state = state();
        random_state.cards.cardsInHand = 1;
        random_state.cards.hand[0] = sts::CardInstance{sts::CardId::SWORD_BOOMERANG};
        stsrl::CombatEnvironment random_environment{std::move(random_state)};
        const auto random_encoding = random_environment.decision().encoding;
        const auto& card = *std::find_if(random_encoding.cards.begin(), random_encoding.cards.end(), [](const auto& token) { return token.card_id == static_cast<int>(sts::CardId::SWORD_BOOMERANG); });
        check(card.target_type == stsrl::TargetType::random_enemy);
        check(std::none_of(random_encoding.card_monster_interactions.begin(), random_encoding.card_monster_interactions.end(), [&](const auto& item) { return random_encoding.cards[item.card_index].card_id == card.card_id; }));
    }

    for (const auto seed : {1ULL, 2ULL, 7ULL, 42ULL, 100ULL, 1234ULL}) {
        auto jaw_env = stsrl::scenarios::jaw_worm(seed);
        const auto jaw_dec = jaw_env.decision();
        check(jaw_dec.encoding.global.card_selection_task == static_cast<int>(sts::CardSelectTask::INVALID));
        for (const auto& a : jaw_dec.encoding.legal_actions) {
            check(a.card_selection_task == static_cast<int>(sts::CardSelectTask::INVALID));
        }
        const auto jaw_actions = jaw_env.search_actions();
        for (const auto& a : jaw_actions) {
            check(a.key.selection_task == static_cast<int>(sts::CardSelectTask::INVALID));
            check(a.key.kind >= 0 && a.key.card_id >= 0 && a.key.cost >= 0 && a.key.target_id >= 0 && a.key.target_hp >= 0);
        }

        auto slime_env = stsrl::scenarios::slime_boss(seed);
        const auto slime_dec = slime_env.decision();
        check(slime_dec.encoding.global.card_selection_task == static_cast<int>(sts::CardSelectTask::INVALID));
        for (const auto& a : slime_dec.encoding.legal_actions) {
            check(a.card_selection_task == static_cast<int>(sts::CardSelectTask::INVALID));
        }
        const auto slime_actions = slime_env.search_actions();
        for (const auto& a : slime_actions) {
            check(a.key.selection_task == static_cast<int>(sts::CardSelectTask::INVALID));
            check(a.key.kind >= 0 && a.key.card_id >= 0 && a.key.cost >= 0 && a.key.target_id >= 0 && a.key.target_hp >= 0);
        }
    }

    auto mcts_environment = stsrl::scenarios::slime_boss(1);
    stsrl::MctsAgent mcts{1, {.simulations = 16, .rollout_limit = 32}};
    const auto result = mcts.search(mcts_environment);
    check(result.root_visits == 16 && std::isfinite(result.root_value) && result.root_value >= -1 && result.root_value <= 1);
    check(std::any_of(result.actions.begin(), result.actions.end(), [&](const auto& x) { return x.execution_index == result.chosen_action && std::isfinite(x.q) && x.q >= -1 && x.q <= 1; }));
    check(!mcts_environment.action_description(result.chosen_action).empty());
}
