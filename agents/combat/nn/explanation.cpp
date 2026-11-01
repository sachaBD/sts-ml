#include "agents/combat/nn/explanation.hpp"
#include "agents/combat/nn/decision.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/record_v4.hpp"
#include "environments/combat/view.hpp"
#include <cmath>
#include <sstream>
#include <string>

namespace stsrl::nn {
namespace {
using Json = nlohmann::json;

std::string key(std::uint64_t k) { return std::to_string(k); }  // JSON numbers lose 64-bit keys in browsers

Fight replay(const Json& request) {
    const auto& start = request.at("start");
    Fight fight{combat_v4::start_battle(start), start.at("seed").get<std::uint64_t>()};
    for (const auto& bits : request.at("moves")) {
        const search::Action move{bits.get<std::uint32_t>()};
        if (fight.over() || !move.isValidAction(fight.state())) throw std::invalid_argument{"illegal move " + bits.dump()};
        fight.play(move);
    }
    return fight;
}

// A node as its query showed it: the board and moves in the world the node was created in.
Json node_json(const search::Query& query, float network) {
    Json edges = Json::array();
    for (const auto move : query.actions) {
        std::ostringstream desc;
        move.printDesc(desc, query.state);
        edges.push_back({{"bits", move.bits}, {"desc", desc.str()}});
    }
    return {{"view", combat::view(query.state)}, {"network", network}, {"edges", edges}};
}
}  // namespace

Json explain(const Json& request, const Network& network) {
    const auto fight = replay(request);
    const auto& state = fight.state();
    Json out{{"steps", fight.steps()}, {"over", fight.over()}, {"observation", key(combat::observation_key(state))},
             {"view", combat::view(state)}};
    if (fight.over()) return out;

    const auto& s = request.at("search");
    search::Gumbel::Settings settings;
    settings.simulations = s.at("sims");
    settings.particles = s.at("particles");
    settings.gumbel_scale = s.at("gumbel_scale");
    settings.rescale_values = s.value("rescale_values", true);
    const double mix = s.at("rollout_mix");

    auto random = fight.decision_random();
    auto search = std::make_unique<search::Gumbel>(state, settings, random());
    const auto& gumbel = *search;
    Decision decision{state, std::move(search), mix, random()};
    Json nodes = Json::array();  // node i answered query i
    while (const auto tokens = decision.pending()) {
        const auto prediction = network(*tokens);
        nodes.push_back(node_json(*gumbel.pending(), prediction.value));
        decision.evaluated(prediction);
    }

    const auto tree = gumbel.tree();
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        const auto& n = tree.nodes[i];
        auto& node = nodes[i];
        node["value"] = n.value;
        node["total"] = n.total;
        // value = (1 - mix) * network + mix * rollout, and a rollout is won (1) or lost (0)
        node["rollout"] = mix > 0 ? Json(std::lround((n.value - (1 - mix) * node["network"].get<double>()) / mix)) : Json();
        for (std::size_t j = 0; j < n.edges.size(); ++j) {
            const auto& e = n.edges[j];
            auto& edge = node["edges"][j];
            edge.update({{"logit", e.logit}, {"prior", e.prior}, {"sigma", e.sigma}, {"improved", e.improved},
                         {"value_sum", e.value_sum}, {"visits", e.visits}, {"children", Json::array()}});
            for (const auto& c : e.children) edge["children"].push_back({{"observation", key(c.observation)}, {"node", c.node}});
        }
    }
    const auto result = decision.result();
    out["decision"] = {{"action", result.action.bits}, {"value", result.value}};
    out["tree"] = {{"gumbel", tree.gumbel}, {"nodes", nodes}};
    return out;
}
}  // namespace stsrl::nn
