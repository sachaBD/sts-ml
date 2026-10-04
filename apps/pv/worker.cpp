// pv_worker encode OUT.parquet: combat_v4 fights (+ search rows) on stdin (JSON lines, a pipe transport) → per-decision
//   training rows in one Parquet shard (apps/pv/shard.hpp).
// pv_worker evaluate MODEL: encoded input batches on stdin → predictions on stdout (parity/debug interface).
// pv_worker play MODEL SIMS [--explore] | teacher SIMS: {fight_id, start} lines on stdin → one result line per fight:
//   {status, fight (combat_v4 fights row; completed only), search: [combat_v4 search rows], stats: [telemetry per search]}.
#include "agents/combat/pv/search.hpp"
#include "agents/combat/search/teacher_search.hpp"
#include "apps/pv/shard.hpp"
#include "environments/combat/environment.hpp"
#include "environments/combat/record_v4.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
namespace pv = stsrl::pv;
using Clock = std::chrono::steady_clock;

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

void encode(const Json& fight, pv::ShardWriter& writer) {
    if (fight.value("status", std::string{"completed"}) != "completed") return;
    const auto actions = fight.at("actions").get<std::vector<std::uint32_t>>();
    // Reject a desynced fight before emitting any rows. This is the canonical replay implementation.
    const auto end = stsrl::combat_v4::replay(fight.at("start"), actions);
    const bool won = end.outcome == sts::Outcome::PLAYER_VICTORY;
    if (won != fight.at("won").get<bool>() || end.player.curHp != fight.at("final_hp").get<int>())
        throw std::runtime_error{"PV: recorded outcome did not replay"};
    std::map<std::size_t, const Json*> searched;
    for (const auto& row : fight.at("search"))
        if (!searched.emplace(row.at("step").get<std::size_t>(), &row).second)
            throw std::runtime_error{"PV: several search rows for one decision"};
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    const float value = pv::CombatObjective::terminal_value(end);
    for (std::size_t step = 0; step < actions.size(); ++step) {
        const auto moves = pv::legal_actions(state);
        std::vector<std::uint32_t> bits;
        for (const auto action : moves) bits.push_back(action.bits);
        std::vector<float> visits(moves.size(), 0);
        double total = 0, matched = 0;
        const Json* row = searched.contains(step) ? searched.at(step) : nullptr;
        if (row)
            for (const auto& child : row->at("children")) {
                const double count = child.at("visits");
                if (!std::isfinite(count) || count < 0) throw std::runtime_error{"PV: invalid recorded visit count"};
                total += count;
                for (std::size_t a = 0; a < bits.size(); ++a) if (bits[a] == child.at("action")) {
                    visits[a] += count; matched += count; break;
                }
            }
        if (matched != total) throw std::runtime_error{"PV: recorded search visits do not match legal actions"};
        if (total > 0) for (auto& v : visits) v /= total;
        const auto inputs = pv::encode(state, moves);
        std::optional<float> root;
        if (row && !row->at("root_value").is_null()) root = row->at("root_value").get<float>();
        writer.add({fight.at("fight_id").get<std::string>(), fight.at("start").at("seed").get<std::uint64_t>(),
                    fight.at("start").at("encounter").get<int>(), int(step), state.turn,
                    won, end.player.curHp, value, root, bits, visits, total > 0, &inputs});
        sts::search::Action{actions[step]}.execute(state);
    }
}

// Result line of one fight. Only a completed fight has a combat_v4 row; a capped one is not a loss.
Json result(const Json& fight, const std::vector<std::uint32_t>& actions, bool done, bool won, int hp,
            const std::string& agent, Json search, Json stats) {
    Json out{{"search", std::move(search)}, {"stats", std::move(stats)}, {"status", done ? "completed" : "capped"}};
    if (done)
        out["fight"] = {{"fight_id", fight.at("fight_id")}, {"version", stsrl::combat_v4::version}, {"start", fight.at("start")},
                        {"actions", actions}, {"explored", std::vector<bool>(actions.size(), false)}, {"won", won},
                        {"final_hp", hp}, {"agent", agent}};
    return out;
}

void check_replay(const Json& fight, const std::vector<std::uint32_t>& actions, const sts::BattleContext& end) {
    const auto rebuilt = stsrl::combat_v4::replay(fight.at("start"), actions);
    if (rebuilt.outcome != end.outcome || rebuilt.player.curHp != end.player.curHp)
        throw std::runtime_error{"PV: recorded actions do not rebuild the played fight"};
}

// PV plays directly on a BattleContext with its own legal-action list (the tree's root edges), so recorded bits replay.
Json play(const Json& fight, pv::Evaluator& evaluator, std::int64_t simulations, bool exploration, double c,
          const std::string& agent) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    pv::SearchSettings settings;
    settings.noise_fraction = exploration ? 0.25 : 0;
    Json search = Json::array(), stats = Json::array();
    std::vector<std::uint32_t> actions;
    while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512) {
        const auto moves = pv::legal_actions(state);
        auto chosen = moves.front();
        if (moves.size() > 1) {
            const auto t = Clock::now();
            // Evaluate the real root exactly once, then construct a fully initialized PV-only tree.
            const auto root_prediction = pv::evaluate_root(evaluator, state);
            auto particles = stsrl::teacher::public_particles(state, 8);
            pv::Search<> tree{state, std::move(particles), root_prediction, settings, pv::PuctScore{c}};
            tree.run([&evaluator](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); }, simulations);
            chosen = tree.selected_action();
            Json children = Json::array();
            double value_sum = 0, visits = 0;
            for (const auto& edge : tree.root().edges) if (edge.visits) {
                children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}, {"value", edge.value_sum / edge.visits}});
                value_sum += edge.value_sum; visits += edge.visits;
            }
            const auto& d = tree.telemetry();
            const double n = tree.simulations();
            search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                              {"root_value", value_sum / visits}, {"simulations", tree.simulations()}, {"children", children}});
            stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", since(t)},
                             {"simulations", tree.simulations()}, {"mean_depth", d.depth_sum / n}, {"max_depth", d.depth_max},
                             {"mean_turns", d.turns_sum / n}, {"max_turns", d.turns_max}, {"nodes", d.nodes}});
        }
        actions.push_back(chosen.bits); chosen.execute(state);
    }
    const bool done = state.outcome != sts::Outcome::UNDECIDED;
    if (done) check_replay(fight, actions, state);
    return result(fight, actions, done, state.outcome == sts::Outcome::PLAYER_VICTORY, state.player.curHp, agent, search, stats);
}

// The guided-rollout MCTS teacher (apps/combat_record/worker.cpp settings). Depth telemetry is PV-only: null here.
Json teach(const Json& fight, const stsrl::teacher::SearchFn& searcher, const std::string& agent) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    stsrl::CombatEnvironment env{std::move(state)};
    Json search = Json::array(), stats = Json::array();
    std::vector<std::uint32_t> actions;
    while (!env.done()) {
        const auto n = env.legal_action_count();
        if (n == 0) throw std::runtime_error{"no legal actions in an undecided fight"};
        std::size_t chosen = 0;
        if (n > 1) {
            const auto t = Clock::now();
            const auto d = stsrl::teacher::search_decision(env, n, searcher, false, stsrl::teacher::particles);
            const double seconds = since(t);
            chosen = d.chosen;
            Json children = Json::array();
            for (const auto& c : d.tried)
                children.push_back({{"action", env.action_bits(c.index)}, {"visits", c.visits}, {"value", c.value}});
            search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                              {"root_value", d.value}, {"simulations", d.used}, {"children", children}});
            stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", seconds},
                             {"simulations", d.used}, {"mean_depth", nullptr}, {"max_depth", nullptr},
                             {"mean_turns", nullptr}, {"max_turns", nullptr}, {"nodes", nullptr}});
        }
        actions.push_back(env.action_bits(chosen)); env.step(chosen);
    }
    check_replay(fight, actions, env.battle());
    return result(fight, actions, true, env.won(), env.player_hp(), agent, search, stats);
}

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (argc == 3 && command == "encode") {
            pv::ShardWriter writer{argv[2]};
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty()) encode(Json::parse(line), writer);
            writer.close();
        } else if (argc == 3 && command == "evaluate") {
            pv::Evaluator evaluator{argv[2]};
            std::string line;
            while (std::getline(std::cin, line)) {
                const auto states = Json::parse(line);
                std::vector<pv::Inputs> batch;
                for (const auto& state : states) {
                    pv::Inputs inputs;
                    for (std::size_t i = 0; i < pv::names.size(); ++i)
                        if (i == 0) inputs[i] = state.at(pv::names[i]).get<std::vector<float>>();
                        else for (const auto& row : state.at(pv::names[i])) {
                            const auto data = row.get<std::vector<float>>();
                            inputs[i].insert(inputs[i].end(), data.begin(), data.end());
                        }
                    batch.push_back(std::move(inputs));
                }
                Json out = Json::array();
                for (const auto& p : evaluator.evaluate(batch)) out.push_back({{"value", p.value}, {"logits", p.logits}});
                std::cout << out.dump() << std::endl;
            }
        } else if (argc >= 4 && command == "play") {
            bool exploration = false;
            double c = pv::PuctScore{}.exploration;
            for (int i = 4; i < argc; ++i) {
                const std::string option = argv[i];
                if (option == "--explore") exploration = true;
                else if (option == "--c" && i + 1 < argc) c = std::stod(argv[++i]);
                else throw std::invalid_argument{"unknown play option " + option};
            }
            pv::Evaluator evaluator{argv[2]};
            const auto simulations = std::stoll(argv[3]);
            if (simulations < 1) throw std::invalid_argument{"PV needs at least one simulation"};
            const std::string agent = "pv model=" + std::filesystem::path{argv[2]}.parent_path().filename().string() + "/" +
                std::filesystem::path{argv[2]}.filename().string() + " sims=" + std::to_string(simulations) +
                " explore=" + (exploration ? "0.25" : "0") + " c=" + std::to_string(c) + " q=minmax particles=8 batch=" +
                std::to_string(pv::SearchSettings{}.batch_size);
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty())
                std::cout << play(Json::parse(line), evaluator, simulations, exploration, c, agent).dump() << std::endl;
        } else if (argc == 3 && command == "teacher") {
            const auto sims = std::stoll(argv[2]);
            const auto searcher = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, stsrl::teacher::particles});
            const std::string agent = "mcts leaf=guided_rollout sims=" + std::to_string(sims) +
                                      " particles=" + std::to_string(stsrl::teacher::particles);
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty())
                std::cout << teach(Json::parse(line), searcher, agent).dump() << std::endl;
        } else {
            std::cerr << "usage: pv_worker encode OUT.parquet | evaluate MODEL.onnx | play MODEL.onnx SIMS [--explore] [--c C] | teacher SIMS (JSONL stdin)\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
