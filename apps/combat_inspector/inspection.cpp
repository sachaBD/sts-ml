#include "apps/combat_inspector/inspection.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace stsrl::inspection {
namespace {
const char* transition_name(search::Gumbel::TransitionKind kind) {
    using enum search::Gumbel::TransitionKind;
    switch (kind) {
    case unknown: return "unknown";
    case deterministic: return "deterministic";
    case chance: return "chance";
    }
    throw std::logic_error{"invalid transition kind"};
}
}  // namespace

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

Json move_json(search::Action action, const sts::BattleContext& state) {
    std::ostringstream description;
    action.printDesc(description, state);
    return {{"bits", action.bits}, {"description", description.str()}};
}

Json edge_json(const search::Gumbel::Graph::Edge& edge) {
    Json children = Json::array();
    for (const auto& child : edge.children)
        children.push_back({{"observation", std::to_string(child.observation)}, {"visits", child.visits},
                            {"bootstrap", child.bootstrap}});
    return {{"action", std::to_string(edge.action)}, {"logit", edge.logit}, {"prior", edge.prior},
            {"sigma", edge.sigma}, {"improved_policy", edge.improved_policy},
            {"q", edge.q ? Json(*edge.q) : Json()}, {"visits", edge.visits}, {"children", children},
            {"transition", transition_name(edge.transition)}};
}

Json graph_json(const search::Gumbel::Graph& graph, std::span<const search::Action> root_actions,
                const sts::BattleContext& state) {
    Json nodes = Json::array();
    for (const auto& node : graph.nodes) {
        Json edges = Json::array();
        for (std::size_t i = 0; i < node.edges.size(); ++i) {
            auto value = edge_json(node.edges[i]);
            if (node.observation == graph.root) value.update(move_json(root_actions[i], state));
            edges.push_back(std::move(value));
        }
        nodes.push_back({{"observation", std::to_string(node.observation)}, {"terminal", node.terminal},
                         {"evaluation_value", node.evaluation_value}, {"backed_value", node.backed_value},
                         {"edges", edges}});
    }
    return {{"root", std::to_string(graph.root)}, {"root_noise", graph.root_noise}, {"nodes", nodes}};
}
}  // namespace stsrl::inspection
