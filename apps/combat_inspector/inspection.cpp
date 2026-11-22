#include "apps/combat_inspector/inspection.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/view.hpp"
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace stsrl::inspector {
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

const char* transition_name(search::Gumbel::TransitionKind kind) {
    using enum search::Gumbel::TransitionKind;
    switch (kind) {
    case unknown: return "unknown";
    case deterministic: return "deterministic";
    case chance: return "chance";
    }
    throw std::logic_error{"invalid transition kind"};
}

Json graph_json(const search::Gumbel::Graph& graph, const search::Result& result, const sts::BattleContext& state,
                const std::unordered_map<std::uint64_t, Json>& views) {
    Json nodes = Json::array();
    for (const auto& node : graph.nodes) {
        Json edges = Json::array();
        const auto found = views.find(node.observation);
        for (std::size_t i = 0; i < node.edges.size(); ++i) {
            const auto& edge = node.edges[i];
            Json children = Json::array();
            for (const auto& child : edge.children)
                children.push_back({{"observation", std::to_string(child.observation)}, {"visits", child.visits},
                                    {"bootstrap", child.bootstrap}});
            Json value{{"action", std::to_string(edge.action)}, {"logit", edge.logit}, {"prior", edge.prior},
                       {"sigma", edge.sigma}, {"improved_policy", edge.improved_policy},
                       {"q", edge.q ? Json(*edge.q) : Json()}, {"visits", edge.visits}, {"children", children},
                       {"transition", transition_name(edge.transition)}};
            if (found != views.end()) {
                for (const auto& move : found->second.at("legal"))
                    if (move.at("action") == std::to_string(edge.action)) value["description"] = move.at("desc");
            }
            if (node.observation == graph.root) {
                const auto action = result.actions.at(i);
                std::ostringstream description;
                action.printDesc(description, state);
                value.update({{"bits", action.bits}, {"description", description.str()}});
            }
            edges.push_back(std::move(value));
        }
        Json entry{{"observation", std::to_string(node.observation)}, {"terminal", node.terminal},
                   {"evaluation_value", node.evaluation_value}, {"backed_value", node.backed_value},
                   {"edges", edges}};
        if (found != views.end()) entry["view"] = found->second;
        nodes.push_back(std::move(entry));
    }
    return {{"root", std::to_string(graph.root)}, {"root_noise", graph.root_noise}, {"nodes", nodes}};
}
}  // namespace

Json inspect(const Json& request, nn::Encoding encoding, const Predict& predict) {
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
    Json response{{"steps", fight.steps()}, {"over", fight.over()},
                  {"observation", std::to_string(combat::observation_key(fight.state()))},
                  {"view", combat::view(fight.state())}};
    if (!fight.over()) {
        const nn::NetworkEvaluator evaluator{encoding};
        auto random = fight.decision_random();
        auto search = config.create(fight.state(), config.sims, random);
        std::unordered_map<std::uint64_t, Json> views;
        views.emplace(combat::observation_key(fight.state()), response["view"]);
        player::Decision decision{std::move(search.search), evaluator};
        for (auto requests = decision.pending(); !requests.empty(); requests = decision.pending()) {
            // Only search nodes, not the guided rollout's intermediate states, belong in the inspector.
            for (const auto& query : search.gumbel.pending()) {
                const auto observation = combat::observation_key(query.state);
                if (!views.contains(observation)) views.emplace(observation, combat::view(query.state));
            }
            decision.evaluated(predict(requests));
        }
        const auto result = decision.result();
        response["decision"] = {{"action", result.action.bits}, {"value", result.value}};
        response["graph"] = graph_json(search.gumbel.graph(), result, fight.state(), views);
    }
    return response;
}
}  // namespace stsrl::inspector
