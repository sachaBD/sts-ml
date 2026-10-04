// pv_worker encode OUT.parquet: combat_v4 fights (+ search rows) on stdin (JSON lines, a pipe transport) → per-decision
//   training rows in one Parquet shard (apps/pv/shard.hpp).
// pv_worker turns: recorded fights on stdin → per-turn DFS measurement rows (pipe transport only).
// pv_worker evaluate MODEL: encoded input batches on stdin → predictions on stdout (parity/debug interface).
// pv_worker play MODEL SIMS [--explore] | teacher SIMS: {fight_id, start} lines on stdin → one result line per fight:
//   {status, fight (combat_v4 fights row; completed only), search: [combat_v4 search rows], stats: [telemetry per search]}.
#include "agents/combat/pv/search.hpp"
#include "agents/combat/pv/turn_search.hpp"
#include "agents/combat/search/teacher_search.hpp"
#include "apps/pv/shard.hpp"
#include "apps/pv/turns.hpp"
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
          double rollout_mix, bool oracle, bool policy_only, bool sample_turns, const std::string& agent) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    pv::SearchSettings settings;
    settings.noise_fraction = exploration ? 0.25 : 0;
    settings.rollout_mix = rollout_mix;
    settings.oracle = oracle;
    std::unique_ptr<pv::Search<>> tree;
    std::mt19937_64 move_random{fight.at("start").at("seed").get<std::uint64_t>()};
    Json search = Json::array(), stats = Json::array();
    std::vector<std::uint32_t> actions;
    while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512) {
        const auto moves = pv::legal_actions(state);
        auto chosen = moves.front();
        if (moves.size() > 1 && policy_only) {
            const auto t = Clock::now();
            const auto prediction = pv::evaluate_root(evaluator, state);
            chosen = moves[std::max_element(prediction.logits.begin(), prediction.logits.end()) - prediction.logits.begin()];
            stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", since(t)},
                             {"simulations", 0}, {"mean_depth", nullptr}, {"max_depth", nullptr},
                             {"mean_turns", nullptr}, {"max_turns", nullptr}, {"nodes", 0}});
        } else if (moves.size() > 1) {
            const auto t = Clock::now();
            // Evaluate the real root exactly once, then construct a fully initialized PV-only tree.
            if (!tree) {
                const auto root_prediction = pv::evaluate_root(evaluator, state);
                auto particles = oracle ? std::vector<sts::BattleContext>{state} : stsrl::teacher::public_particles(state, 8);
                tree = std::make_unique<pv::Search<>>(state, std::move(particles), root_prediction, settings, pv::PuctScore{c});
            }
            tree->run([&evaluator](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); }, simulations);
            chosen = sample_turns && state.turn < 2 ? tree->sampled_action(move_random) : tree->selected_action();
            Json children = Json::array();
            double value_sum = 0, visits = 0;
            for (const auto& edge : tree->root().edges) if (edge.visits) {
                children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}, {"value", edge.value_sum / edge.visits}});
                value_sum += edge.value_sum; visits += edge.visits;
            }
            const auto& d = tree->telemetry();
            const double n = tree->simulations();
            search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                              {"root_value", value_sum / visits}, {"simulations", tree->simulations()}, {"children", children}});
            stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", since(t)},
                             {"simulations", tree->simulations()}, {"mean_depth", d.depth_sum / n}, {"max_depth", d.depth_max},
                             {"mean_turns", d.turns_sum / n}, {"max_turns", d.turns_max}, {"nodes", d.nodes}, {"turns_hist", d.turns_hist}});
        }
        actions.push_back(chosen.bits); chosen.execute(state);
        if (tree && (!oracle || state.outcome != sts::Outcome::UNDECIDED || !tree->advance(chosen, state))) tree.reset();
    }
    const bool done = state.outcome != sts::Outcome::UNDECIDED;
    if (done) check_replay(fight, actions, state);
    return result(fight, actions, done, state.outcome == sts::Outcome::PLAYER_VICTORY, state.player.curHp, agent, search, stats);
}

// Turn-search evaluation mode deliberately emits no invented per-action policy targets.
Json play_turn(const Json& fight, pv::Evaluator& evaluator, std::size_t budget, double c,
               pv::TurnSearchCaps caps, const std::string& agent) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    std::unique_ptr<pv::TurnSearch> tree;
    std::vector<std::uint32_t> actions;
    Json search = Json::array(), stats = Json::array(), turns = Json::array();
    while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512) {
        const auto started = Clock::now(); const auto step = actions.size(); const int turn = state.turn;
        if (!tree) tree = std::make_unique<pv::TurnSearch>(state, caps, c);
        const auto sequence = tree->decide(budget, [&](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); });
        auto telemetry = tree->stats();
        if (!telemetry.fallback) {
            for (auto bits : sequence) {
                if (actions.size() >= 512) break;
                const sts::search::Action chosen{bits};
                if (!chosen.isValidAction(state)) throw std::runtime_error{"turn search: live sequence illegal"};
                actions.push_back(bits); chosen.execute(state);
            }
            if (!tree->advance(state)) tree.reset();
        } else {
            tree.reset();
            pv::SearchSettings settings; settings.oracle = true;
            std::unique_ptr<pv::Search<>> fallback;
            do {
                const auto moves = pv::legal_actions(state); auto chosen = moves.front();
                if (moves.size() > 1) {
                    const auto t = Clock::now();
                    if (!fallback) {
                        const auto prediction = pv::evaluate_root(evaluator, state);
                        ++telemetry.network_calls; ++telemetry.evaluated_states;
                        fallback = std::make_unique<pv::Search<>>(state, std::vector<sts::BattleContext>{state}, prediction, settings, pv::PuctScore{c});
                    }
                    fallback->run([&](std::span<const pv::Inputs> batch) {
                        ++telemetry.network_calls; telemetry.evaluated_states += batch.size();
                        return evaluator.evaluate(batch);
                    }, 800);
                    chosen = fallback->selected_action();
                    Json children = Json::array(); double value = 0, visits = 0;
                    for (const auto& edge : fallback->root().edges) if (edge.visits) {
                        children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}, {"value", edge.value_sum / edge.visits}});
                        value += edge.value_sum; visits += edge.visits;
                    }
                    const auto& d = fallback->telemetry(); const double n = fallback->simulations();
                    search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                        {"root_value", value / visits}, {"simulations", fallback->simulations()}, {"children", children}});
                    stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", since(t)},
                        {"simulations", fallback->simulations()}, {"mean_depth", d.depth_sum / n}, {"max_depth", d.depth_max},
                        {"mean_turns", d.turns_sum / n}, {"max_turns", d.turns_max}, {"nodes", d.nodes}, {"turns_hist", d.turns_hist}});
                }
                actions.push_back(chosen.bits); chosen.execute(state);
                if (fallback && (state.outcome != sts::Outcome::UNDECIDED || !fallback->advance(chosen, state))) fallback.reset();
            } while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512 &&
                     !(state.turn != turn && state.inputState == sts::InputState::PLAYER_NORMAL));
        }
        turns.push_back({{"fight_id", fight.at("fight_id")}, {"step", step}, {"turn", turn}, {"seconds", since(started)},
            {"expansions", telemetry.expansions}, {"network_calls", telemetry.network_calls}, {"evaluated_states", telemetry.evaluated_states},
            {"root_children", telemetry.root_children}, {"max_depth", telemetry.max_depth}, {"mean_leaf_depth", telemetry.mean_leaf_depth},
            {"fallback", telemetry.fallback}, {"reason", telemetry.reason}, {"time_overshoot", telemetry.time_overshoot}, {"reused", telemetry.reused}});
    }
    const bool done = state.outcome != sts::Outcome::UNDECIDED;
    if (done) check_replay(fight, actions, state);
    auto out = result(fight, actions, done, state.outcome == sts::Outcome::PLAYER_VICTORY, state.player.curHp, agent, search, stats);
    out["turn_stats"] = std::move(turns);
    return out;
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
        } else if (argc == 2 && command == "turns") {
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty()) pv::measure_turns(Json::parse(line));
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
            bool exploration = false, oracle = false, policy_only = false, sample_turns = false, turn_search = false;
            pv::TurnSearchCaps caps;
            double c = pv::PuctScore{}.exploration, rollout_mix = 0;
            for (int i = 4; i < argc; ++i) {
                const std::string option = argv[i];
                if (option == "--explore") exploration = true;
                else if (option == "--oracle") oracle = true;
                else if (option == "--turn-search") turn_search = true;
                else if (option == "--turn-max-sequences" && i + 1 < argc) caps.max_sequences = std::stoull(argv[++i]);
                else if (option == "--turn-root-max-children" && i + 1 < argc) caps.root_max_children = std::stoull(argv[++i]);
                else if (option == "--turn-max-children" && i + 1 < argc) caps.max_children = std::stoull(argv[++i]);
                else if (option == "--turn-max-seconds" && i + 1 < argc) caps.max_seconds = std::stod(argv[++i]);
                else if (option == "--policy-only") policy_only = true;
                else if (option == "--sample-turns") sample_turns = true;
                else if (option == "--c" && i + 1 < argc) c = std::stod(argv[++i]);
                else if (option == "--rollout-mix" && i + 1 < argc) rollout_mix = std::stod(argv[++i]);
                else throw std::invalid_argument{"unknown play option " + option};
            }
            if (turn_search && (!oracle || exploration || policy_only || sample_turns || rollout_mix != 0))
                throw std::invalid_argument{"--turn-search requires --oracle and evaluation-only settings"};
            pv::Evaluator evaluator{argv[2]};
            const auto simulations = std::stoll(argv[3]);
            if (simulations < 1) throw std::invalid_argument{"PV needs at least one simulation"};
            std::string agent = "pv model=" + std::filesystem::path{argv[2]}.parent_path().filename().string() + "/" +
                std::filesystem::path{argv[2]}.filename().string() + " sims=" + std::to_string(simulations) +
                " explore=" + (exploration ? "0.25" : "0") + " c=" + std::to_string(c) + " q=minmax particles=" + (oracle ? "1" : "8") + " batch=" +
                std::to_string(pv::SearchSettings{}.batch_size) +
                (rollout_mix > 0 ? " rollout_mix=" + std::to_string(rollout_mix) : "") +
                (oracle ? " oracle=1 reuse=1" : "") + (policy_only ? " policy_only=1" : "") +
                (sample_turns ? " sample_turns=2" : "");
            if (turn_search) agent += " turn_search=1 E=" + std::to_string(simulations) +
                " caps=seq" + std::to_string(caps.max_sequences) + "/root" + std::to_string(caps.root_max_children) +
                "/node" + std::to_string(caps.max_children) +
                "/" + std::to_string(caps.max_seconds) + "s/512a/256MiB T=10 fallback_sims=800";
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty()) {
                const auto fight = Json::parse(line);
                std::cout << (turn_search ? play_turn(fight, evaluator, simulations, c, caps, agent) :
                    play(fight, evaluator, simulations, exploration, c, rollout_mix, oracle, policy_only, sample_turns, agent)).dump() << std::endl;
            }
        } else if (argc == 3 && command == "teacher") {
            const auto sims = std::stoll(argv[2]);
            const auto searcher = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, stsrl::teacher::particles});
            const std::string agent = "mcts leaf=guided_rollout sims=" + std::to_string(sims) +
                                      " particles=" + std::to_string(stsrl::teacher::particles);
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty())
                std::cout << teach(Json::parse(line), searcher, agent).dump() << std::endl;
        } else {
            std::cerr << "usage: pv_worker encode OUT.parquet | turns | evaluate MODEL.onnx | play MODEL.onnx SIMS [--explore] [--oracle] [--turn-search] [--policy-only] [--sample-turns] [--c C] [--rollout-mix L] | teacher SIMS (JSONL stdin)\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
