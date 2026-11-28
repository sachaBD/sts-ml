// Replay fight starts from stdin with the frozen CPU checkpoint; emit one inspection record per decision.
// With STEP, play the recorded settings up to that decision, search it with the given settings, emit it, and stop.
// EXPLORE > 0 draws that decision's move from the search policy with SEED; 0 plays the leader.
#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/search/filter/filter.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/view.hpp"
#include <iostream>
#include <optional>

using namespace stsrl;
using Json = nlohmann::json;
namespace {
struct Settings {
    int cap = 160, worlds = 16;
    double price = .0005;
    search::FilterMode mode = search::FilterMode::full;
    double explore = 0;
    std::uint64_t seed = 0;
};
search::FilterMode parse_mode(const std::string& s) {
    if (s == "full") return search::FilterMode::full;
    if (s == "root_only") return search::FilterMode::root_only;
    if (s == "fixed_uncertainty") return search::FilterMode::fixed_uncertainty;
    throw std::invalid_argument{"mode must be full, root_only or fixed_uncertainty"};
}
}
int main(int argc, char** argv) try {
    if (argc != 2 && argc != 9)
        throw std::invalid_argument{"usage: filter_replay MODEL_DIR [STEP CAP WORLDS PRICE MODE EXPLORE SEED] < starts.jsonl"};
    std::optional<int> target;
    Settings variant;
    if (argc == 9) {
        target = std::stoi(argv[2]);
        variant = {std::stoi(argv[3]), std::stoi(argv[4]), std::stod(argv[5]), parse_mode(argv[6]), std::stod(argv[7]),
                   std::stoull(argv[8])};
    }
    auto net = nn::load_network(argv[1], nn::Backends::cpu);
    nn::NetworkEvaluator evaluator{net.encoding};
    for (std::string line; std::getline(std::cin, line);) {
        const auto row = Json::parse(line);
        player::Fight fight{combat::start_battle(row.at("start")), row.at("start").at("seed").get<std::uint64_t>()};
        player::Answers previous;
        while (!fight.over()) {
            const auto& state = fight.state();
            const auto step = fight.steps();
            if (target && step > *target) break;
            const Settings use = target && step == *target ? variant : Settings{};
            search::FilterTrace trace;
            const auto stream = state.seed ^ (std::uint64_t(state.turn) << 32);
            player::Decision decision{search::filter_search(state, use.cap, use.worlds, use.price, use.mode, stream,
                                                            &trace, use.explore, use.seed),
                                      evaluator, std::move(previous)};
            while (!decision.pending().empty()) {
                std::vector<infra::Input> inputs;
                for (const auto& tokens : decision.pending()) inputs.push_back({&tokens, {}});
                decision.evaluated(net.backends.back()->evaluate(inputs));
            }
            const auto result = decision.result();
            previous = std::move(decision).answers();
            auto view = combat::view(state);
            Json nodes = Json::array();
            for (const auto& n : trace.nodes) {
                nodes.push_back({{"id", n.id}, {"parent", n.parent}, {"parent_edge", n.parent_edge},
                                 {"world", n.world}, {"born", n.born}, {"turn", n.turn},
                                 {"key", std::to_string(n.key)}, {"terminal", n.terminal}, {"cycle", n.cycle},
                                 {"cached", n.cached}, {"nn_value", n.nn_value}, {"prior_sd", n.prior_sd},
                                 {"logits", n.logits}, {"thinking", n.thinking}, {"actions", n.actions},
                                 {"edge_chance", n.edge_chance}});
            }
            Json steps = Json::array();
            for (const auto& s : trace.steps) {
                steps.push_back({{"parent", s.parent}, {"edge", s.edge}, {"world", s.world}, {"child", s.child},
                                 {"spent", s.spent}, {"world_sample", s.world_sample}, {"voi", s.voi},
                                 {"weight", s.weight}, {"gap", s.gap}, {"innovation", s.innovation},
                                 {"root_q", s.root_q}});
            }
            fight.play(result.action);
            Json out{{"fight", row.at("fight_id")}, {"step", step}, {"view", view}, {"chosen", result.action.bits},
                     {"opening", trace.opening.thinking}, {"value", trace.opening.value},
                     {"root_cached", trace.opening.cached}, {"initial_q", trace.initial_q}, {"spent", trace.spent},
                     {"stop", trace.stop}, {"last_voi", trace.last_voi}, {"nodes", nodes}, {"steps", steps},
                     {"next_turn", fight.state().turn}, {"next_hp", fight.state().player.curHp},
                     {"next_energy", fight.state().player.energy}};
            if (!target || step == *target) std::cout << out.dump() << '\n';
            if (target && step == *target) break;
        }
    }
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
