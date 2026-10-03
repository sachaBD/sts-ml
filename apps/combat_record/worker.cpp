// Play the first fights of seeded Ironclad act 1 runs and print them as combat_v4 rows (one JSON per line).
//
//   combat_record_worker SEED MAX_FIGHTS SIMULATIONS COLLECTION SEARCH_STATS
//
// SEARCH_STATS: on (also print each searched decision's search row) or off (fights rows only).
// stdout, one line per fight: {"fight": <fights row>, "search": [<search rows>]}.
// stderr, last line: {"timing": {...}} seconds spent searching, building / printing JSON, and rebuild-checking.
//
// SimpleAgent plays out of combat (environments/overworld/act1_run); guided-rollout MCTS (8 particles) plays
// every fight with SIMULATIONS per decision; forced moves are played without a search. Every fight is rebuilt
// with combat_v4::replay (sts_lightspeed only) and checked against the played fight's complete final battle state
// before it is printed.
#include "agents/combat/search/teacher_search.hpp"
#include "environments/combat/battle_snapshot.hpp"
#include "environments/combat/environment.hpp"
#include "environments/combat/record_v4.hpp"
#include "environments/overworld/act1_run.hpp"
#include "constants/CharacterClasses.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using Json = nlohmann::json;
namespace teacher = stsrl::teacher;
using Clock = std::chrono::steady_clock;

namespace {
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "usage: combat_record_worker SEED MAX_FIGHTS SIMULATIONS COLLECTION on|off\n"; return 2; }
    const std::string stats_arg = argv[5];
    if (stats_arg != "on" && stats_arg != "off") { std::cerr << "SEARCH_STATS must be on or off\n"; return 2; }
    const bool search_stats = stats_arg == "on";
    const std::uint64_t seed = std::stoull(argv[1]);
    const int max_fights = std::stoi(argv[2]);
    const std::int64_t sims = std::stoll(argv[3]);
    const std::string collection = argv[4];
    const auto search = teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, teacher::particles});
    const std::string agent = "mcts leaf=guided_rollout sims=" + std::to_string(sims) +
                              " particles=" + std::to_string(teacher::particles);
    double search_s = 0, record_s = 0, check_s = 0;

    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, 20};
    int index = 0;
    stsrl::act1::play(game, [&](const sts::BattleContext& start, const Json&) {
        // act1::play calls battle.init(game) right before this: game holds the pre-battle fields.
        auto t = Clock::now();
        const Json start_record = stsrl::combat_v4::start_json(game);
        const std::string fight_id = collection + ":" + std::to_string(seed) + ":" + std::to_string(index++);
        record_s += since(t);
        stsrl::CombatEnvironment env{start};
        std::vector<std::uint32_t> actions;
        std::vector<bool> explored;
        Json search_rows = Json::array();
        while (!env.done()) {
            const auto n = env.legal_action_count();
            if (n == 0) throw std::runtime_error{"no legal actions in an undecided fight"};
            std::size_t chosen = 0;
            if (n > 1) {
                t = Clock::now();
                const auto d = teacher::search_decision(env, n, search, false, teacher::particles);
                search_s += since(t);
                chosen = d.chosen;
                if (search_stats) {
                    t = Clock::now();
                    Json children = Json::array();
                    for (const auto& c : d.tried)
                        children.push_back({{"action", env.action_bits(c.index)}, {"visits", c.visits}, {"value", c.value}});
                    search_rows.push_back({{"fight_id", fight_id}, {"step", actions.size()}, {"agent", agent},
                                           {"root_value", d.value}, {"simulations", d.used}, {"children", children}});
                    record_s += since(t);
                }
            }
            actions.push_back(env.action_bits(chosen));
            explored.push_back(false);
            env.step(chosen);
        }
        const auto& end = env.battle();
        t = Clock::now();
        const auto rebuilt = stsrl::combat_v4::replay(start_record, actions);
        if (stsrl::battle_snapshot(rebuilt) != stsrl::battle_snapshot(end))
            throw std::runtime_error{"combat_v4 rebuild does not match the played fight"};
        check_s += since(t);
        t = Clock::now();
        const Json fight{{"fight_id", fight_id}, {"version", stsrl::combat_v4::version}, {"start", start_record},
                         {"actions", actions}, {"explored", explored},
                         {"won", end.outcome == sts::Outcome::PLAYER_VICTORY}, {"final_hp", end.player.curHp},
                         {"agent", agent}};
        std::cout << Json{{"fight", fight}, {"search", search_rows}}.dump() << '\n';
        record_s += since(t);
        return end;
    }, max_fights);
    std::cerr << Json{{"timing", {{"search", search_s}, {"record", record_s}, {"check", check_s}}}}.dump() << '\n';
    return 0;
}
