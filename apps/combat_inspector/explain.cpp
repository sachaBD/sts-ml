#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include "infra/workers/actors.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
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

const char* transition_name(search::Gumbel::TransitionKind kind) {
    using enum search::Gumbel::TransitionKind;
    switch (kind) {
    case unknown: return "unknown";
    case deterministic: return "deterministic";
    case chance: return "chance";
    }
    throw std::logic_error{"invalid transition kind"};
}

Json graph_json(const search::Gumbel::Graph& graph, const search::Result& result, const sts::BattleContext& state) {
    Json nodes = Json::array();
    for (const auto& node : graph.nodes) {
        Json edges = Json::array();
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
            if (node.observation == graph.root) {
                const auto action = result.actions.at(i);
                std::ostringstream description;
                action.printDesc(description, state);
                value.update({{"bits", action.bits}, {"description", description.str()}});
            }
            edges.push_back(std::move(value));
        }
        nodes.push_back({{"observation", std::to_string(node.observation)}, {"terminal", node.terminal},
                         {"evaluation_value", node.evaluation_value}, {"backed_value", node.backed_value},
                         {"edges", edges}});
    }
    return {{"root", std::to_string(graph.root)}, {"root_noise", graph.root_noise}, {"nodes", nodes}};
}
search::SearchConfig search_config(const Json& settings) {
    for (const auto* name : {"sims", "particles", "considered_actions"}) {
        const auto& value = settings.at(name);
        if (!value.is_number_integer() || value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max())
            throw std::invalid_argument{std::string{"search."} + name + " must fit an integer"};
    }
    if (!settings.at("world_draw").is_number_unsigned()) throw std::invalid_argument{"search.world_draw must be unsigned"};
    const search::SearchConfig config{settings.at("sims"), settings.at("particles"), settings.at("gumbel_scale"),
                                      settings.at("rollout_mix"), settings.at("considered_actions"),
                                      settings.at("world_draw").get<std::uint64_t>()};
    config.validate();
    return config;
}

// Response lines, one per request, written as each finishes.
class Responses {
public:
    void write(Json response) {
        std::lock_guard lock{mutex_};
        std::cout << response.dump() << '\n';
        if (!std::cout) throw std::runtime_error{"failed to write a response"};
    }

private:
    std::mutex mutex_;
};

// One request: its replayed fight and, unless the fight is over, the decision's search.
class Inspection final : public infra::Episode {
public:
    Inspection(std::size_t index, const Json& request, const nn::NetworkEvaluator& evaluator, Responses& responses)
        : index_{index}, fight_{replay(request.at("start"), request.at("moves"))}, responses_{responses} {
        const auto config = search_config(request.at("search"));
        if (fight_.over()) return;
        auto random = fight_.decision_random();
        auto search = config.create(fight_.state(), config.sims, random);
        gumbel_ = &search.gumbel;
        decision_.emplace(std::move(search.search), evaluator);
    }

    std::span<const infra::Tokens> advance() override {
        return decision_ ? decision_->pending() : std::span<const infra::Tokens>{};
    }
    void evaluated(std::span<const infra::Prediction> predictions) override { decision_->evaluated(predictions); }

    void finish() override {
        Json response{{"index", index_}, {"steps", fight_.steps()}, {"over", fight_.over()}};
        if (decision_) {
            const auto result = decision_->result();
            response["decision"] = {{"action", result.action.bits}, {"value", result.value}};
            response["network_requests"] = decision_->sent();
            response["graph"] = graph_json(gumbel_->graph(), result, fight_.state());
        }
        responses_.write(std::move(response));
    }
    std::string label() const override { return "request " + std::to_string(index_); }

private:
    std::size_t index_;
    player::Fight fight_;
    Responses& responses_;
    const search::Gumbel* gumbel_ = nullptr;  // owned by decision_'s search
    std::optional<player::Decision> decision_;
};
}  // namespace

int main(int argc, char** argv) try {
    CLI::App app{"Searches each request line on stdin ({start, moves, search}) with one network and prints one response "
                 "line per request, tagged with its line index, in completion order.", "combat_explain"};
    std::filesystem::path models;
    bool cpu = false;
    app.add_option("MODELS", models, "exported model directory")->required();
    app.add_flag("--cpu", cpu, "evaluate on the CPU only");
    CLI11_PARSE(app, argc, argv);
    std::vector<Json> requests;
    for (std::string line; std::getline(std::cin, line);)
        if (!line.empty()) requests.push_back(Json::parse(line));
    const auto network = nn::load_network(models, cpu ? nn::Backends::cpu : nn::Backends::all);
    const nn::NetworkEvaluator evaluator{network.encoding};
    std::vector<infra::Backend*> backends;
    for (const auto& backend : network.backends) backends.push_back(backend.get());
    Responses responses;
    std::atomic<std::size_t> next{0};
    infra::run_actors({}, [&]() -> std::unique_ptr<infra::Episode> {
        const auto i = next++;
        return i < requests.size() ? std::make_unique<Inspection>(i, requests[i], evaluator, responses) : nullptr;
    }, backends);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "combat_explain: " << error.what() << '\n';
    return 1;
}
