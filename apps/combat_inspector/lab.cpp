// A search lab session: searches stay open between commands, and the caller picks every simulation's root move and
// world. One JSON command per stdin line, one JSON response per stdout line; a failed command answers {"error"}.
//   {"op": "open", "trees": [{"start", "moves", "draw"}]}  -> {"trees": [{"id", "steps", "over", "moves", "root"}]}
//   {"op": "simulate", "picks": {"<id>": [[move, world], ...]}}  -> {"roots": {"<id>": root}}  (one wave per tree)
//   {"op": "graph", "tree": id}  -> {"graph"}
//   {"op": "close", "trees": [id]}  -> {}
// A fight already over opens no tree (id null). Opening evaluates each root; every network call of a command is
// one batch.
#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "apps/combat_inspector/inspection.hpp"
#include "environments/combat/public.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>

namespace {
using namespace stsrl;
using namespace stsrl::inspection;

class Network {
public:
    Network(const std::filesystem::path& models, bool cpu)
        : network_{nn::load_network(models, cpu ? nn::Backends::cpu : nn::Backends::all)}, evaluator_{network_.encoding} {}

    // Answers every pending query of the searches, batched per backend, until none is pending.
    // Returns the requests each search sent.
    std::vector<int> answer(const std::vector<search::Gumbel*>& searches) {
        std::vector<int> sent(searches.size());
        for (;;) {
            std::vector<infra::Tokens> requests;
            std::vector<std::size_t> owner;
            for (std::size_t s = 0; s < searches.size(); ++s)
                for (const auto& query : searches[s]->pending()) {
                    requests.push_back(evaluator_.request(query));
                    owner.push_back(s);
                    ++sent[s];
                }
            if (requests.empty()) return sent;
            const auto predictions = evaluate(requests);
            std::size_t next = 0;
            for (std::size_t s = 0; s < searches.size(); ++s) {
                std::vector<search::Evaluation> evaluations;
                for (; next < owner.size() && owner[next] == s; ++next)
                    evaluations.push_back(nn::NetworkEvaluator::answer(predictions[next]));
                if (!evaluations.empty()) searches[s]->evaluated(evaluations);
            }
        }
    }

private:
    std::vector<infra::Prediction> evaluate(const std::vector<infra::Tokens>& requests) {
        std::vector<infra::Prediction> out(requests.size());
        std::vector<std::vector<std::size_t>> routed(network_.backends.size());
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto backend = std::ranges::find_if(network_.backends, [&](const auto& b) { return b->accepts(requests[i]); });
            if (backend == network_.backends.end()) throw std::runtime_error{"no backend accepts a request"};
            routed[backend - network_.backends.begin()].push_back(i);
        }
        for (std::size_t b = 0; b < routed.size(); ++b) {
            auto& backend = *network_.backends[b];
            const auto& rows = routed[b];
            for (std::size_t first = 0; first < rows.size(); first += backend.max_batch()) {
                const auto last = std::min(rows.size(), first + backend.max_batch());
                std::vector<std::vector<std::uint16_t>> packed(last - first);
                std::vector<infra::Input> inputs;
                for (std::size_t r = first; r < last; ++r) {
                    backend.prepare(requests[rows[r]], packed[r - first]);
                    inputs.push_back({&requests[rows[r]], packed[r - first]});
                }
                auto predictions = backend.evaluate(inputs);
                for (std::size_t r = first; r < last; ++r) out[rows[r]] = std::move(predictions[r - first]);
            }
        }
        return out;
    }

    nn::Network network_;
    nn::NetworkEvaluator evaluator_;
};

struct Tree {
    player::Fight fight;
    std::unique_ptr<search::Gumbel> search;
    int requests = 0;
};

std::unique_ptr<search::Gumbel> steered(const player::Fight& fight, std::uint64_t draw) {
    search::Gumbel::Settings settings;
    settings.simulations = 0;
    settings.particles = 1;
    settings.world_draw = draw;
    auto random = fight.decision_random();
    return std::make_unique<search::Gumbel>(fight.state(), settings, random());
}

Json root_json(const Tree& tree) {
    const auto root = tree.search->root();
    Json moves = Json::array();
    for (std::size_t i = 0; i < root.edges.size(); ++i) {
        const auto& edge = root.edges[i];
        Json worlds = Json::array();
        for (const auto& w : root.worlds[i]) worlds.push_back({{"world", w.world}, {"visits", w.visits}, {"q", w.q}});
        moves.push_back({{"logit", edge.logit}, {"prior", edge.prior}, {"sigma", edge.sigma},
                         {"improved_policy", edge.improved_policy}, {"q", edge.q ? Json(*edge.q) : Json()},
                         {"visits", edge.visits}, {"worlds", worlds}});
    }
    return {{"value", root.value}, {"requests", tree.requests}, {"moves", moves}};
}

class Lab {
public:
    Lab(const std::filesystem::path& models, bool cpu) : network_{models, cpu} {}

    Json run(const Json& command) {
        const auto op = command.at("op").get<std::string>();
        if (op == "open") return open(command.at("trees"));
        if (op == "simulate") return simulate(command.at("picks"));
        if (op == "graph") {
            const auto& tree = at(command.at("tree"));
            return {{"graph", graph_json(tree.search->graph(), legal(tree), tree.fight.state())}};
        }
        if (op == "close") {
            for (const auto& id : command.at("trees")) trees_.erase(id.get<int>());
            return Json::object();
        }
        throw std::invalid_argument{"unknown op " + op};
    }

private:
    static std::vector<search::Action> legal(const Tree& tree) { return combat::legal_actions(tree.fight.state()); }

    Tree& at(const Json& id) {
        const auto found = trees_.find(id.get<int>());
        if (found == trees_.end()) throw std::invalid_argument{"no open tree " + id.dump()};
        return found->second;
    }

    Json open(const Json& requests) {
        std::vector<std::optional<int>> ids;
        std::vector<search::Gumbel*> searches;
        for (const auto& request : requests) {
            auto fight = replay(request.at("start"), request.at("moves"));
            if (fight.over()) { ids.emplace_back(); continue; }
            auto search = steered(fight, request.at("draw").get<std::uint64_t>());
            searches.push_back(search.get());
            ids.emplace_back(next_);
            trees_.emplace(next_++, Tree{std::move(fight), std::move(search)});
        }
        const auto sent = network_.answer(searches);
        Json out = Json::array();
        std::size_t s = 0;
        for (const auto& id : ids) {
            if (!id) { out.push_back({{"id", nullptr}, {"over", true}}); continue; }
            auto& tree = trees_.at(*id);
            tree.requests += sent[s++];
            Json moves = Json::array();
            for (const auto action : legal(tree)) moves.push_back(move_json(action, tree.fight.state()));
            out.push_back({{"id", *id}, {"steps", tree.fight.steps()}, {"over", false}, {"moves", moves},
                           {"root", root_json(tree)}});
        }
        return {{"trees", out}};
    }

    Json simulate(const Json& picks) {
        std::vector<Tree*> chosen;
        std::vector<search::Gumbel*> searches;
        for (const auto& [id, wave] : picks.items()) {
            auto& tree = at(Json(std::stoi(id)));
            std::vector<search::Gumbel::Pick> steer;
            for (const auto& pick : wave) steer.push_back({pick.at(0).get<std::size_t>(), pick.at(1).get<int>()});
            tree.search->simulate(steer);
            chosen.push_back(&tree);
            searches.push_back(tree.search.get());
        }
        const auto sent = network_.answer(searches);
        Json roots = Json::object();
        std::size_t s = 0;
        for (const auto& [id, wave] : picks.items()) {
            chosen[s]->requests += sent[s];
            roots[id] = root_json(*chosen[s++]);
        }
        return {{"roots", roots}};
    }

    Network network_;
    std::map<int, Tree> trees_;
    int next_ = 0;
};
}  // namespace

int main(int argc, char** argv) try {
    CLI::App app{"A search lab session: JSON commands on stdin, one JSON response per line on stdout (see lab.cpp).",
                 "combat_lab"};
    std::filesystem::path models;
    bool cpu = false;
    app.add_option("MODELS", models, "exported model directory")->required();
    app.add_flag("--cpu", cpu, "evaluate on the CPU only");
    CLI11_PARSE(app, argc, argv);
    Lab lab{models, cpu};
    for (std::string line; std::getline(std::cin, line);) {
        if (line.empty()) continue;
        Json response;
        try {
            response = lab.run(Json::parse(line));
        } catch (const std::exception& error) {
            response = {{"error", error.what()}};
        }
        std::cout << response.dump() << std::endl;
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "combat_lab: " << error.what() << '\n';
    return 1;
}
