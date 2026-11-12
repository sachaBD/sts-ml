#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using namespace stsrl;

player::Fight replay(const Json& start, const Json& moves) {
    player::Fight fight{combat::start_battle(start), start.at("seed").get<std::uint64_t>()};
    if (!moves.is_array()) throw std::invalid_argument{"moves must be an array of action bits"};
    for (const auto& bits : moves) {
        if (!bits.is_number_integer() || bits < 0 || bits > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument{"action bits must be unsigned 32-bit integers"};
        if (fight.over()) throw std::invalid_argument{"move after fight ended"};
        const search::Action action{bits.get<std::uint32_t>()};
        const auto legal = combat::legal_actions(fight.state());
        if (std::ranges::find(legal, action) == legal.end())
            throw std::invalid_argument{"illegal move " + bits.dump()};
        fight.play(action);
    }
    return fight;
}

Json graph_json(const search::Gumbel::Graph& graph, const search::Result& result, const sts::BattleContext& state) {
    Json nodes = Json::array();
    for (const auto& node : graph.nodes) {
        Json edges = Json::array();
        for (std::size_t i = 0; i < node.edges.size(); ++i) {
            const auto& edge = node.edges[i];
            Json children = Json::array();
            for (const auto& child : edge.children)
                children.push_back({{"observation", std::to_string(child.observation)}, {"visits", child.visits}});
            Json value{{"action", std::to_string(edge.action)}, {"logit", edge.logit}, {"prior", edge.prior},
                       {"sigma", edge.sigma}, {"improved_policy", edge.improved_policy},
                       {"q", edge.q ? Json(*edge.q) : Json()}, {"visits", edge.visits}, {"children", children}};
            if (node.observation == graph.root) {
                const auto action = result.actions.at(i);
                std::ostringstream description;
                action.printDesc(description, state);
                value.update({{"bits", action.bits}, {"description", description.str()}});
            }
            edges.push_back(std::move(value));
        }
        nodes.push_back({{"observation", std::to_string(node.observation)}, {"terminal", node.terminal},
                         {"evaluation_value", node.evaluation_value}, {"mean_value", node.mean_value},
                         {"visits", node.visits}, {"edges", edges}});
    }
    return {{"root", std::to_string(graph.root)}, {"root_noise", graph.root_noise}, {"nodes", nodes}};
}
}  // namespace

int main() try {
    const auto request = Json::parse(std::cin);
    const auto& settings = request.at("search");
    for (const auto* name : {"sims", "particles"}) {
        const auto& value = settings.at(name);
        if (!value.is_number_integer() || value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max())
            throw std::invalid_argument{std::string{"search."} + name + " must fit an integer"};
    }
    const search::SearchConfig config{settings.at("sims"), settings.at("particles"),
                                      settings.at("gumbel_scale"), settings.at("rollout_mix")};
    config.validate();
    const auto fight = replay(request.at("start"), request.at("moves"));
    Json response{{"steps", fight.steps()}, {"over", fight.over()}};
    if (!fight.over()) {
        const auto network = nn::load_network(request.at("checkpoint").get<std::string>(), nn::Backends::cpu);
        const nn::NetworkEvaluator evaluator{network.encoding};
        auto random = fight.decision_random();
        auto search = config.create(fight.state(), random);
        player::Decision decision{std::move(search.search), evaluator};
        for (auto requests = decision.pending(); !requests.empty(); requests = decision.pending()) {
            std::vector<infra::Input> inputs;
            for (const auto& tokens : requests) inputs.push_back({&tokens, {}});
            decision.evaluated(network.backends.front()->evaluate(inputs));
        }
        const auto result = decision.result();
        response["decision"] = {{"action", result.action.bits}, {"value", result.value}};
        response["graph"] = graph_json(search.gumbel.graph(), result, fight.state());
    }
    std::cout << response.dump() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "combat_explain: " << error.what() << '\n';
    return 1;
}
