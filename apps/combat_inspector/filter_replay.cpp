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
Json key_strings(const std::vector<std::uint64_t>& keys) {
    Json out = Json::array();
    for (const auto k : keys) out.push_back(std::to_string(k));
    return out;
}
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
        player::Fight fight{combat::start_battle(row.at("start")),row.at("start").at("seed").get<std::uint64_t>()};
        player::Answers previous;
        while (!fight.over()) {
            const auto& state = fight.state();
            const auto step = fight.steps();
            if (target && step > *target) break;
            const Settings use = target && step == *target ? variant : Settings{};
            search::FilterTrace trace;
            const auto stream = state.seed ^ (std::uint64_t(state.turn) << 32);
            player::Decision decision{search::filter_search(state,use.cap,use.worlds,use.price,use.mode,stream,&trace,use.explore,use.seed),evaluator,std::move(previous)};
            while (!decision.pending().empty()) {
                std::vector<infra::Input> inputs;
                for (const auto& tokens : decision.pending()) inputs.push_back({&tokens,{}});
                decision.evaluated(net.backends.back()->evaluate(inputs));
            }
            const auto result = decision.result();
            previous = std::move(decision).answers();
            auto view = combat::view(state);
            Json trials = Json::array();
            for (const auto& t : trace.trials) trials.push_back({{"arm",t.arm},{"world",t.world},{"spent",t.spent},{"interior",t.interior},
                {"rate",t.rate},{"candidate",t.candidate},{"reference",t.reference},{"before",t.mean_before},{"after",t.mean_after},
                {"sd0",t.innovation_before},{"sd1",t.innovation_after},{"scores",t.scores},{"line",t.line},{"reference_line",t.reference_line},
                {"keys",key_strings(t.keys)},{"reference_keys",key_strings(t.reference_keys)}});
            Json nodes = Json::array();
            for (const auto& n : trace.nodes) nodes.push_back({{"key",std::to_string(n.key)},{"value",n.value},{"logits",n.logits},
                {"thinking",n.thinking},{"actions",n.actions},{"previous",n.previous}});
            fight.play(result.action);
            Json out{{"fight",row.at("fight_id")},{"step",step},{"view",view},{"chosen",result.action.bits},{"trials",trials},
                {"opening",trace.opening.thinking},{"value",trace.opening.value},{"root_cached",trace.opening.cached},
                {"scores",trace.initial_scores},{"reserve",trace.reserve},{"spent",trace.spent},{"stop",trace.stop},{"last_rate",trace.last_rate},
                {"next_turn",fight.state().turn},{"next_hp",fight.state().player.curHp},{"next_energy",fight.state().player.energy},
                {"root_key",std::to_string(trace.root)},{"nodes",nodes}};
            if (!target || step == *target) std::cout << out.dump() << '\n';
            if (target && step == *target) break;
        }
    }
} catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
