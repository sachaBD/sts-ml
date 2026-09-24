#include "scenarios/act1_run.hpp"

#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <cctype>

namespace stsrl::act1 {
namespace {

// easy / hard: act 1 hallway pools; event: a fight started by an event ("?" room).
std::string category(const sts::GameContext& game, sts::MonsterEncounter encounter) {
    namespace pool = sts::MonsterEncounterPool;
    if (game.curRoom == sts::Room::BOSS) return "boss";
    if (game.curRoom == sts::Room::ELITE) return "elite";
    if (game.curRoom == sts::Room::MONSTER) {
        const auto act = game.act - 1;
        const auto in = [&](const auto* list, int count) { return std::find(list, list + count, encounter) != list + count; };
        if (in(pool::weakEnemies[act], pool::weakCount[act])) return "easy";
        if (in(pool::strongEnemies[act], pool::strongCount[act])) return "hard";
    }
    return "event";
}

std::string encounter_name(sts::MonsterEncounter encounter) {
    std::string name = sts::monsterEncounterEnumNames[static_cast<int>(encounter)];
    for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return name;
}

nlohmann::json fight_columns(const sts::GameContext& game, const sts::BattleContext& battle, int fight_index) {
    return {{"run_seed", game.seed}, {"episode_id", static_cast<std::int64_t>(game.seed) * 100 + fight_index},
            {"fight_index", fight_index}, {"act", game.act}, {"floor", game.floorNum},
            {"encounter", encounter_name(battle.encounter)}, {"category", category(game, battle.encounter)},
            {"ascension", game.ascension}, {"starting_hp", battle.player.curHp},
            {"starting_max_hp", battle.player.maxHp}};
}

}  // namespace

Result play(sts::GameContext& game, const FightFn& fight, int max_fights) {
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    int fights = 0;
    bool boss_beaten = false;
    while (game.outcome == sts::GameOutcome::UNDECIDED && game.act == 1 && !boss_beaten && fights != max_fights) {
        if (game.screenState != sts::ScreenState::BATTLE) {
            agent.stepOutOfCombat(game);
            continue;
        }
        sts::BattleContext battle;
        battle.init(game);
        const auto end = fight(battle, fight_columns(game, battle, fights++));
        end.exitBattle(game);
        boss_beaten = game.curRoom == sts::Room::BOSS && game.outcome == sts::GameOutcome::UNDECIDED;
    }
    const auto status = boss_beaten ? "act_complete" : game.outcome == sts::GameOutcome::PLAYER_LOSS ? "died" : "stopped";
    return {status, game.floorNum, fights};
}

}  // namespace stsrl::act1
