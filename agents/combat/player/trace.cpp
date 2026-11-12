#include "agents/combat/player/trace.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

namespace stsrl::player {
namespace {
double rounded(double x, double scale) { return std::round(x * scale) / scale; }
}  // namespace

nlohmann::json trace(const search::Gumbel::Graph& graph, const search::Result& result, const sts::BattleContext& state) {
    std::unordered_map<std::uint64_t, std::size_t> index;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) index.emplace(graph.nodes[i].observation, i);
    auto nodes = nlohmann::json::array();
    for (const auto& node : graph.nodes) {
        auto logits = nlohmann::json::array();
        for (const auto& edge : node.edges) logits.push_back(rounded(edge.logit, 1e3));
        nodes.push_back({rounded(node.evaluation_value, 1e4), node.terminal, logits});
    }
    auto actions = nlohmann::json::array(), descriptions = nlohmann::json::array();
    for (const auto action : result.actions) {
        std::ostringstream description;
        action.printDesc(description, state);
        actions.push_back(action.bits);
        descriptions.push_back(description.str());
    }
    auto backups = nlohmann::json::array();
    for (const auto& backup : graph.backups) {
        auto path = nlohmann::json::array();
        for (const auto& step : backup.path) {
            const auto& edges = graph.nodes[index.at(step.observation)].edges;
            const auto edge = std::ranges::find(edges, step.action, &search::Gumbel::Graph::Edge::action) - edges.begin();
            path.push_back(edge);
            path.push_back(index.at(step.reached));
        }
        backups.push_back(std::move(path));
    }
    return {{"action", result.action.bits}, {"value", result.value}, {"root", index.at(graph.root)},
            {"root_actions", actions}, {"root_descriptions", descriptions}, {"root_noise", graph.root_noise},
            {"nodes", nodes}, {"backups", backups}};
}
}  // namespace stsrl::player
