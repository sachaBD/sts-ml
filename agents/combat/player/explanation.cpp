#include "agents/combat/player/explanation.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/search/config.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace stsrl::player {
namespace {
using Json = nlohmann::json;
std::string key(std::uint64_t value) { return std::to_string(value); }
}

Json explain(const Json& request, nn::Encoding encoding, const Network& network) {
    const auto& start = request.at("start");
    Fight fight{combat::start_battle(start), start.at("seed").get<std::uint64_t>()};
    for (const auto& bits : request.at("moves")) {
        const auto value = bits.get<std::int64_t>();
        if (value < 0 || value > UINT32_MAX) throw std::invalid_argument{"action bits out of range"};
        const search::Action action{std::uint32_t(value)};
        const auto legal = combat::legal_actions(fight.state());
        if (fight.over() || std::ranges::find(legal, action) == legal.end())
            throw std::invalid_argument{"illegal move " + bits.dump()};
        fight.play(action);
    }
    const auto& s = request.at("search");
    const search::SearchConfig config{s.at("sims"), s.at("particles"), s.at("gumbel_scale"), s.at("rollout_mix")};
    config.validate();
    Json out{{"steps", fight.steps()}, {"over", fight.over()},
             {"observation", key(combat::observation_key(fight.state()))}};
    if (fight.over()) return out;

    search::Gumbel::Settings settings;
    settings.simulations = config.sims;
    settings.particles = config.particles;
    settings.gumbel_scale = config.gumbel_scale;
    auto random = fight.decision_random();
    auto gumbel = std::make_unique<search::Gumbel>(fight.state(), settings, random());
    const auto& graph = *gumbel;
    std::unique_ptr<search::CombatSearch> search = std::move(gumbel);
    const auto rollout_seed = random();
    if (config.rollout_mix > 0)
        search = search::with_rollouts(std::move(search), fight.state(), config.rollout_mix, rollout_seed);
    const nn::NetworkEvaluator evaluator{encoding};
    Decision decision{std::move(search), evaluator};
    Json queries = Json::object();
    while (const auto tokens = decision.pending()) {
        const auto query = *graph.pending();
        const auto prediction = network(*tokens);
        Json actions = Json::array();
        for (const auto action : query.actions) {
            std::ostringstream description;
            action.printDesc(description, query.state);
            actions.push_back({{"bits", action.bits}, {"description", description.str()}});
        }
        queries[key(combat::observation_key(query.state))] = {{"network", prediction.value}, {"actions", actions}};
        decision.evaluated(prediction);
    }
    const auto tree = graph.tree();
    Json nodes = Json::array();
    for (const auto& n : tree.nodes) {
        const auto observation = key(n.observation);
        Json node{{"observation", observation}, {"value", n.value}, {"mean_value", n.mean_value},
                  {"visits", n.visits}, {"terminal", n.edges.empty()}, {"edges", Json::array()}};
        if (queries.contains(observation)) node["network"] = queries.at(observation).at("network");
        for (std::size_t i = 0; i < n.edges.size(); ++i) {
            const auto& e = n.edges[i];
            Json edge = queries.at(observation).at("actions").at(i);
            edge.update({{"action", key(e.action)}, {"logit", e.logit}, {"prior", e.prior}, {"sigma", e.sigma},
                         {"improved", e.improved}, {"q", e.visits ? Json(e.value) : Json()},
                         {"visits", e.visits}, {"children", Json::array()}});
            for (const auto& child : e.children)
                edge["children"].push_back({{"observation", key(child.observation)}, {"visits", child.visits}});
            node["edges"].push_back(std::move(edge));
        }
        nodes.push_back(std::move(node));
    }
    const auto result = decision.result();
    out["decision"] = {{"action", result.action.bits}, {"value", result.value}};
    out["tree"] = {{"root", key(tree.root)}, {"gumbel", tree.gumbel}, {"nodes", nodes}};
    return out;
}
}  // namespace stsrl::player
