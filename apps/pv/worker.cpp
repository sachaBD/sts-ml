// pv_worker encode: combat_v4 fights (+ optional search rows) on stdin → per-decision training rows on stdout.
// pv_worker evaluate MODEL: encoded input batches on stdin → predictions on stdout (parity/debug interface).
#include "agents/combat/pv/search.hpp"
#include "agents/combat/search/teacher_leaves.hpp"
#include "environments/combat/record_v4.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
namespace pv = stsrl::pv;

Json input_json(const pv::Inputs& input) {
    Json result;
    for (std::size_t i = 0; i < pv::names.size(); ++i) {
        if (i == 0) result[pv::names[i]] = input[i];
        else {
            result[pv::names[i]] = Json::array();
            for (std::size_t start = 0; start < input[i].size(); start += pv::widths[i])
                result[pv::names[i]].push_back(std::vector<float>{input[i].begin() + start, input[i].begin() + start + pv::widths[i]});
        }
    }
    return result;
}

void encode(const Json& fight) {
    if (fight.value("status", std::string{"completed"}) != "completed") return;
    const auto actions = fight.at("actions").get<std::vector<std::uint32_t>>();
    // Reject a desynced fight before emitting any rows. This is the canonical replay implementation.
    const auto end = stsrl::combat_v4::replay(fight.at("start"), actions);
    const bool won = end.outcome == sts::Outcome::PLAYER_VICTORY;
    if (won != fight.at("won").get<bool>() || end.player.curHp != fight.at("final_hp").get<int>())
        throw std::runtime_error{"PV: recorded outcome did not replay"};
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    for (std::size_t step = 0; step < actions.size(); ++step) {
        const auto moves = pv::legal_actions(state);
        std::vector<std::uint32_t> bits;
        for (const auto action : moves) bits.push_back(action.bits);
        std::vector<double> visits(moves.size(), 0);
        double total = 0, matched = 0;
        for (const auto& row : fight.value("search", Json::array())) if (row.at("step") == step)
            for (const auto& child : row.at("children")) {
                const double count = child.at("visits");
                if (!std::isfinite(count) || count < 0) throw std::runtime_error{"PV: invalid recorded visit count"};
                total += count;
                for (std::size_t a = 0; a < bits.size(); ++a) if (bits[a] == child.at("action")) {
                    visits[a] += count; matched += count; break;
                }
            }
        if (matched != total) throw std::runtime_error{"PV: recorded search visits do not match legal actions"};
        if (total > 0) for (auto& v : visits) v /= total;
        const double value = pv::CombatObjective::terminal_value(end);
        std::cout << Json{{"contract", pv::contract}, {"fight_id", fight.at("fight_id")}, {"seed", fight.at("start").at("seed")}, {"step", step},
                         {"inputs", input_json(pv::encode(state, moves))}, {"moves", bits}, {"value_target", value},
                         {"policy_target", visits}, {"has_policy", total > 0 && matched == total}}.dump() << '\n';
        sts::search::Action{actions[step]}.execute(state);
    }
}

// Returned facts can be fed straight back to `encode` for expert iteration. Capped fights are not losses.
Json play(const Json& fight, pv::Evaluator& evaluator, std::int64_t simulations, bool exploration) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    stsrl::CombatEnvironment env{std::move(state)};
    if (simulations < 1) throw std::invalid_argument{"PV needs at least one simulation"};
    pv::SearchSettings settings;
    settings.noise_fraction = exploration ? 0.25 : 0;
    Json trace = Json::array(), actions = Json::array();
    while (!env.done() && env.battle().turn < 50 && actions.size() < 512) {
        const auto count = env.legal_action_count();
        std::size_t chosen = 0;
        if (count > 1) {
            // Evaluate the real root exactly once, then construct a fully initialized PV-only tree.
            const auto root_prediction = pv::evaluate_root(evaluator, env.battle());
            auto particles = stsrl::teacher::public_particles(env.battle(), 8);
            pv::Search<> tree{env.battle(), std::move(particles), root_prediction, settings};
            tree.run([&evaluator](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); }, simulations);

            const auto selected = tree.selected_action();
            chosen = count;
            for (std::size_t i = 0; i < count; ++i) {
                if (env.action_bits(i) == selected.bits) { chosen = i; break; }
            }
            if (chosen == count) throw std::runtime_error{"PV: selected action is not legal in the real battle"};
            Json children = Json::array();
            for (const auto& edge : tree.root().edges) if (edge.visits) {
                children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}});
            }
            trace.push_back({{"step", actions.size()}, {"children", children}});
        }
        actions.push_back(env.action_bits(chosen)); env.step(chosen);
    }
    return {{"fight_id", fight.at("fight_id")}, {"start", fight.at("start")}, {"actions", actions},
            {"search", trace}, {"won", env.won()}, {"final_hp", env.player_hp()},
            {"status", env.done() ? "completed" : "capped"}};
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string{argv[1]} == "encode") {
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty()) encode(Json::parse(line));
        } else if (argc == 3 && std::string{argv[1]} == "evaluate") {
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
        } else if ((argc == 4 || argc == 5) && std::string{argv[1]} == "play") {
            const bool exploration = argc == 5 && std::string{argv[4]} == "--explore";
            if (argc == 5 && !exploration) throw std::invalid_argument{"unknown play option"};
            pv::Evaluator evaluator{argv[2]};
            const auto simulations = std::stoll(argv[3]);
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty())
                std::cout << play(Json::parse(line), evaluator, simulations, exploration).dump() << std::endl;
        } else {
            std::cerr << "usage: pv_worker encode | evaluate MODEL.onnx | play MODEL.onnx SIMS [--explore] (JSONL stdin)\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
