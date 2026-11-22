#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "apps/combat_inspector/inspection.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/view.hpp"
#include "infra/workers/actors.hpp"
#include <CLI/CLI.hpp>
#include <atomic>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace {
using namespace stsrl;
using namespace stsrl::inspection;

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
        : index_{index}, fight_{replay(request.at("start"), request.at("moves"))},
          include_views_{request.value("views", false)}, responses_{responses} {
        const auto config = search_config(request.at("search"));
        if (include_views_) views_.emplace(combat::observation_key(fight_.state()), combat::view(fight_.state()));
        if (fight_.over()) return;
        auto random = fight_.decision_random();
        auto search = config.create(fight_.state(), config.sims, random);
        gumbel_ = &search.gumbel;
        decision_.emplace(std::move(search.search), evaluator);
    }

    std::span<const infra::Tokens> advance() override {
        if (!decision_) return {};
        if (include_views_) {
            for (const auto& query : gumbel_->pending()) {
                const auto obs = combat::observation_key(query.state);
                if (!views_.contains(obs)) views_.emplace(obs, combat::view(query.state));
            }
        }
        return decision_->pending();
    }
    void evaluated(std::span<const infra::Prediction> predictions) override { decision_->evaluated(predictions); }

    void finish() override {
        const auto obs = combat::observation_key(fight_.state());
        Json response{{"index", index_}, {"steps", fight_.steps()}, {"over", fight_.over()}};
        if (include_views_) {
            response["observation"] = std::to_string(obs);
            response["view"] = views_.at(obs);
        }
        if (decision_) {
            const auto result = decision_->result();
            const auto graph = gumbel_->graph();
            if (include_views_) fill_terminal_views(*gumbel_, graph, views_);
            response["decision"] = {{"action", result.action.bits}, {"value", result.value}};
            response["network_requests"] = decision_->sent();
            response["graph"] = graph_json(graph, result.actions, fight_.state(), views_);
        }
        responses_.write(std::move(response));
    }
    std::string label() const override { return "request " + std::to_string(index_); }

private:
    std::size_t index_;
    player::Fight fight_;
    bool include_views_;
    Responses& responses_;
    const search::Gumbel* gumbel_ = nullptr;  // owned by decision_'s search
    std::optional<player::Decision> decision_;
    std::unordered_map<std::uint64_t, Json> views_;
};
}  // namespace

int main(int argc, char** argv) try {
    CLI::App app{"Searches each request line on stdin ({start, moves, search, views?}) with one network and prints "
                 "one response line per request, tagged with its line index, in completion order.", "combat_explain"};
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
