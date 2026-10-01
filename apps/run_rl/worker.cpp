// Real-run RL worker (slop_docs/run-rl/README.md). Long-running; JSON lines on stdin/stdout.
// Plays real Ironclad act 1 runs: the MCTS teacher (guided-rollout leaves, no recording) plays every fight,
// SimpleAgent plays everything out of combat EXCEPT card rewards, which are asked of Python.
//
// Python -> worker, one line per run:
//   {"seed": S, "ascension": A, "simulations": {easy, hard, elite, event, boss}}
// worker -> Python, at every card reward (answered by one line {"choice": i}; i = len(options) means skip):
//   {"type": "pick", "state": macro_sim::state_json, "boss": name, "options": [card...], "simple": i}
//   simple = SimpleAgent's choice for this reward (the baseline policy).
// worker -> Python, when the run ends:
//   {"type": "done", "seed", "boss", "status": died | act_complete, "floor", "fights", "final_hp", "seconds",
//    "steps": [...]}   steps in play order:
//     {"kind": "start", "state"}                                         before the first floor
//     {"kind": "fight", "state", "encounter", "category", "won", "hp_before"}   state after exitBattle
//     {"kind": "pick", "state", "options", "choice", "simple"}           state BEFORE the pick
#include "agents/teacher_search.hpp"
#include "apps/common/game_state.hpp"
#include "apps/common/macro_sim.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "sim/search/GameAction.h"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::json;
using sts::GameContext;
namespace teacher = stsrl::teacher;

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string encounter_name(sts::MonsterEncounter e) { return lower(sts::monsterEncounterEnumNames[static_cast<int>(e)]); }

// As scenarios/act1_run.cpp (budget category of a fight).
std::string category(const GameContext& game, sts::MonsterEncounter encounter) {
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

Json card_json(const sts::Card& c) {
    return {{"card_id", static_cast<int>(c.id)}, {"upgraded", c.getUpgraded()}, {"misc", c.misc},
            {"name", lower(sts::cardEnumStrings[static_cast<int>(c.id)])}};
}

// A card-reward decision is pending: SimpleAgent would take potions / relics / keys first (as card_search).
bool card_decision(const GameContext& gc) {
    if (gc.screenState != sts::ScreenState::REWARDS) return false;
    const auto& r = gc.info.rewardsContainer;
    if (r.potionCount > 0 && gc.potionCount < gc.potionCapacity) return false;
    return r.relicCount == 0 && !r.sapphireKey && !r.emeraldKey && r.cardRewardCount > 0;
}

const auto& last_reward(const GameContext& gc) {
    const auto& r = gc.info.rewardsContainer;
    return r.cardRewards[r.cardRewardCount - 1];
}

void take(GameContext& gc, int choice) {
    const auto& r = gc.info.rewardsContainer;
    if (choice < static_cast<int>(last_reward(gc).size()))
        sts::search::GameAction(sts::search::GameAction::RewardsActionType::CARD, r.cardRewardCount - 1, choice).execute(gc);
    else
        sts::search::GameAction(sts::search::GameAction::RewardsActionType::SKIP).execute(gc);
}

// SimpleAgent's choice at this reward: run its card step on a copy and see which offered card entered the deck.
int simple_choice(const GameContext& gc) {
    GameContext copy = gc;
    sts::search::SimpleAgent agent;
    agent.curGameContext = &copy;
    agent.stepCardReward(copy);
    const auto& offered = last_reward(gc);
    const int n = static_cast<int>(offered.size());
    if (copy.deck.size() == gc.deck.size()) return n;  // skipped
    std::map<int, int> before;
    for (const auto& c : gc.deck.cards) ++before[static_cast<int>(c.id)];
    for (const auto& c : copy.deck.cards)
        if (before[static_cast<int>(c.id)]-- <= 0)
            for (int i = 0; i < n; ++i)
                if (offered[i].id == c.id) return i;
    return n;
}

Json ask(const Json& message) {
    std::cout << message.dump() << '\n' << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) throw std::runtime_error{"stdin closed while waiting for a pick"};
    return Json::parse(line);
}

Json play_run(const Json& job) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto seed = job.at("seed").get<std::uint64_t>();
    const auto& sims = job.at("simulations");
    for (const auto* key : {"easy", "hard", "elite", "event", "boss"})
        if (sims.at(key).get<std::int64_t>() < 1) throw std::invalid_argument{"simulations must be positive"};
    const teacher::Leaf leaf{"guided_rollout", 0, 0};
    teacher::validate(leaf, false);

    GameContext gc{sts::CharacterClass::IRONCLAD, seed, job.at("ascension").get<int>()};
    sts::search::SimpleAgent agent;
    agent.curGameContext = &gc;
    Json steps = Json::array();
    steps.push_back({{"kind", "start"}, {"state", stsrl::macro_sim::state_json(gc)}});
    int fights = 0;
    bool boss_beaten = false;
    while (gc.outcome == sts::GameOutcome::UNDECIDED && gc.act == 1 && !boss_beaten) {
        if (gc.screenState != sts::ScreenState::BATTLE) {
            if (card_decision(gc)) {
                Json options = Json::array();
                for (const auto& c : last_reward(gc)) options.push_back(card_json(c));
                const int simple = simple_choice(gc);
                auto state = stsrl::macro_sim::state_json(gc);
                const auto reply = ask({{"type", "pick"}, {"state", state}, {"boss", encounter_name(gc.boss)},
                                        {"options", options}, {"simple", simple}});
                const int choice = reply.at("choice").get<int>();
                if (choice < 0 || choice > static_cast<int>(options.size())) throw std::invalid_argument{"bad choice"};
                take(gc, choice);
                steps.push_back({{"kind", "pick"}, {"state", std::move(state)}, {"options", options},
                                 {"choice", choice}, {"simple", simple}});
                continue;
            }
            agent.stepOutOfCombat(gc);
            continue;
        }
        sts::BattleContext battle;
        battle.init(gc);
        const auto cat = category(gc, battle.encounter);
        const int hp_before = battle.player.curHp;
        const auto search = teacher::leaf_search(leaf, nullptr, {sims.at(cat).get<std::int64_t>(), teacher::particles});
        const auto end = teacher::play_fight(battle, search, false, teacher::particles, false);
        end.exitBattle(gc);
        ++fights;
        const bool won = gc.outcome != sts::GameOutcome::PLAYER_LOSS;
        boss_beaten = gc.curRoom == sts::Room::BOSS && won;
        steps.push_back({{"kind", "fight"}, {"state", stsrl::macro_sim::state_json(gc)},
                         {"encounter", encounter_name(battle.encounter)}, {"category", cat}, {"won", won},
                         {"hp_before", hp_before}});
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {{"type", "done"}, {"seed", seed}, {"boss", encounter_name(gc.boss)},
            {"status", boss_beaten ? "act_complete" : "died"}, {"floor", gc.floorNum}, {"fights", fights},
            {"final_hp", gc.curHp}, {"seconds", seconds}, {"steps", steps}};
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        try {
            std::cout << play_run(Json::parse(line)).dump() << '\n' << std::flush;
        } catch (const std::exception& e) {
            std::cout << Json{{"type", "error"}, {"message", e.what()}}.dump() << '\n' << std::flush;
            return 1;
        }
    }
    return 0;
}
