#include "apps/combat_inspector/inspection.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/view.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

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

int checked_int(const Json& settings, const char* name) {
    const auto& value = settings.at(name);
    if (!value.is_number_integer() || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max())
        throw std::invalid_argument{std::string{"search."} + name + " must fit an integer"};
    return value.get<int>();
}

Json decision_json(const search::Result& result) {
    return {{"action", result.action.bits}, {"value", result.value}};
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

search::SearchConfig search_config(const Json& settings) {
    const int sims = checked_int(settings, "sims");
    const int particles = checked_int(settings, "particles");
    const int considered = settings.contains("considered_actions")
        ? checked_int(settings, "considered_actions")
        : search::Gumbel::Settings{}.considered_actions;
    std::uint64_t world_draw = search::Gumbel::Settings{}.world_draw;
    if (settings.contains("world_draw")) {
        if (!settings.at("world_draw").is_number_unsigned())
            throw std::invalid_argument{"search.world_draw must be unsigned"};
        world_draw = settings.at("world_draw").get<std::uint64_t>();
    }
    const search::SearchConfig config{sims, particles, settings.at("gumbel_scale"), settings.at("rollout_mix"),
                                      considered, world_draw, true};
    config.validate();
    return config;
}

Json move_json(search::Action action, const sts::BattleContext& state) {
    std::ostringstream description;
    action.printDesc(description, state);
    return {{"bits", action.bits}, {"description", description.str()}};
}

Json edge_json(const search::Gumbel::Graph::Edge& edge) {
    Json children = Json::array();
    for (const auto& child : edge.children) {
        Json worlds = Json::array();
        for (const auto& w : child.worlds) worlds.push_back(w.world);
        children.push_back({{"observation", std::to_string(child.observation)}, {"visits", child.visits},
                            {"bootstrap", child.bootstrap}, {"worlds", std::move(worlds)}});
    }
    return {{"action", std::to_string(edge.action)}, {"logit", edge.logit}, {"prior", edge.prior},
            {"sigma", edge.sigma}, {"improved_policy", edge.improved_policy},
            {"q", edge.q ? Json(*edge.q) : Json()}, {"visits", edge.visits}, {"children", children},
            {"transition", transition_name(edge.transition)}};
}

void fill_terminal_views(const search::Gumbel& gumbel, const search::Gumbel::Graph& graph,
                         std::unordered_map<std::uint64_t, Json>& views) {
    std::size_t missing = 0;
    std::unordered_map<std::uint64_t, const search::Gumbel::Graph::Node*> by_obs;
    for (const auto& node : graph.nodes) {
        by_obs.emplace(node.observation, &node);
        if (!views.contains(node.observation)) ++missing;
    }
    if (!missing) return;
    struct Entry { std::uint64_t observation; std::vector<std::size_t> path; int world; };
    std::deque<Entry> queue;
    std::set<std::pair<std::uint64_t, int>> seen;
    const auto* root = by_obs.at(graph.root);
    for (std::size_t e = 0; e < root->edges.size(); ++e)
        for (const auto& child : root->edges[e].children)
            for (const auto& w : child.worlds)
                if (seen.emplace(child.observation, w.world).second)
                    queue.push_back({child.observation, {e}, w.world});
    while (!queue.empty() && missing) {
        auto current = std::move(queue.front());
        queue.pop_front();
        if (!views.contains(current.observation)) {
            views.emplace(current.observation, combat::view(gumbel.state(current.path, current.world)));
            --missing;
        }
        const auto* node = by_obs.at(current.observation);
        for (std::size_t e = 0; e < node->edges.size(); ++e)
            for (const auto& child : node->edges[e].children) {
                if (child.bootstrap) continue;
                if (std::ranges::find(child.worlds, current.world, &search::Gumbel::Graph::WorldVisits::world) ==
                    child.worlds.end())
                    continue;
                if (!seen.emplace(child.observation, current.world).second) continue;
                auto next_path = current.path;
                next_path.push_back(e);
                queue.push_back({child.observation, std::move(next_path), current.world});
            }
    }
}

Json graph_json(const search::Gumbel::Graph& graph, std::span<const search::Action> root_actions,
                const sts::BattleContext& state, const std::unordered_map<std::uint64_t, Json>& views) {
    Json nodes = Json::array();
    for (const auto& node : graph.nodes) {
        Json edges = Json::array();
        const auto found = views.find(node.observation);
        for (std::size_t i = 0; i < node.edges.size(); ++i) {
            auto value = edge_json(node.edges[i]);
            if (found != views.end()) {
                for (const auto& move : found->second.at("legal"))
                    if (move.at("action") == std::to_string(node.edges[i].action)) value["description"] = move.at("desc");
            }
            if (node.observation == graph.root) {
                if (i >= root_actions.size()) throw std::out_of_range{"graph_json: root_actions shorter than root edges"};
                value.update(move_json(root_actions[i], state));
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

namespace {
// One world's own expectimax: the best of the edges its simulations took, each the mean of the links it took.
double world_value(const search::Gumbel::Graph::Node& node, int world,
                   const std::unordered_map<std::uint64_t, const search::Gumbel::Graph::Node*>& by_obs,
                   std::unordered_map<std::uint64_t, double>& memo) {
    if (const auto found = memo.find(node.observation); found != memo.end()) return found->second;
    std::optional<double> best;
    for (const auto& edge : node.edges) {
        double sum = 0;
        int count = 0;
        for (const auto& child : edge.children) {
            if (std::ranges::find(child.worlds, world, &search::Gumbel::Graph::WorldVisits::world) ==
                child.worlds.end())
                continue;
            const auto& next = *by_obs.at(child.observation);
            sum += child.bootstrap ? next.evaluation_value : world_value(next, world, by_obs, memo);
            ++count;
        }
        if (count) best = std::max(best.value_or(-INFINITY), sum / count);
    }
    return memo[node.observation] = best.value_or(node.evaluation_value);
}
}  // namespace

Json root_json(const search::Gumbel::Graph& graph, int requests) {
    std::unordered_map<std::uint64_t, const search::Gumbel::Graph::Node*> by_obs;
    for (const auto& node : graph.nodes) by_obs.emplace(node.observation, &node);
    const auto& root = *by_obs.at(graph.root);
    Json moves = Json::array();
    for (const auto& edge : root.edges) {
        std::map<int, int> visits_by_world;
        for (const auto& child : edge.children)
            for (const auto& w : child.worlds) visits_by_world[w.world] += w.visits;
        Json worlds = Json::array();
        for (const auto& [world, visits] : visits_by_world) {
            std::unordered_map<std::uint64_t, double> memo;
            double sum = 0;
            int links = 0;
            for (const auto& child : edge.children) {
                if (std::ranges::find(child.worlds, world, &search::Gumbel::Graph::WorldVisits::world) ==
                    child.worlds.end())
                    continue;
                const auto& next = *by_obs.at(child.observation);
                sum += child.bootstrap ? next.evaluation_value : world_value(next, world, by_obs, memo);
                ++links;
            }
            worlds.push_back({{"world", world}, {"visits", visits}, {"q", sum / links}});
        }
        moves.push_back({{"logit", edge.logit}, {"prior", edge.prior}, {"sigma", edge.sigma},
                         {"improved_policy", edge.improved_policy}, {"q", edge.q ? Json(*edge.q) : Json()},
                         {"visits", edge.visits}, {"worlds", std::move(worlds)}});
    }
    return {{"value", root.backed_value}, {"requests", requests}, {"moves", std::move(moves)}};
}

struct Session::Impl {
    struct Tree {
        player::Fight fight;
        std::unique_ptr<search::CombatSearch> search;
        search::Gumbel* gumbel = nullptr;  // owned by `search`, even when wrapped in RolloutBlend
        std::unordered_map<std::uint64_t, Json> views;
        int requests = 0;
    };

    nn::NetworkEvaluator evaluator;
    Predict predict;
    std::map<int, Tree> trees;
    int next_id = 0;

    Impl(nn::Encoding encoding, Predict predict_fn) : evaluator{encoding}, predict{std::move(predict_fn)} {}

    static std::vector<search::Action> legal(const Tree& tree) { return combat::legal_actions(tree.fight.state()); }

    Tree& at(const Json& id) {
        const auto found = trees.find(id.get<int>());
        if (found == trees.end()) throw std::invalid_argument{"no open tree " + id.dump()};
        return found->second;
    }

    // Answers every pending query of `active`, caching each node's public board view and batching across trees.
    void answer(const std::vector<Tree*>& active) {
        for (;;) {
            std::vector<infra::Tokens> requests;
            std::vector<std::size_t> owner;
            for (std::size_t t = 0; t < active.size(); ++t) {
                for (const auto& query : active[t]->search->pending()) {
                    const auto obs = combat::observation_key(query.state);
                    if (!active[t]->views.contains(obs)) active[t]->views.emplace(obs, combat::view(query.state));
                    requests.push_back(evaluator.request(query));
                    owner.push_back(t);
                    ++active[t]->requests;
                }
            }
            if (requests.empty()) return;
            const auto predictions = predict(requests);
            if (predictions.size() != requests.size())
                throw std::runtime_error{"inspector: one prediction per network request"};
            std::size_t next = 0;
            for (std::size_t t = 0; t < active.size(); ++t) {
                std::vector<search::Evaluation> evaluations;
                for (; next < owner.size() && owner[next] == t; ++next)
                    evaluations.push_back(nn::NetworkEvaluator::answer(predictions[next]));
                if (!evaluations.empty()) active[t]->search->evaluated(evaluations);
            }
        }
    }

    Json tree_graph(Tree& tree, const search::Gumbel::Graph& graph) {
        fill_terminal_views(*tree.gumbel, graph, tree.views);
        return graph_json(graph, legal(tree), tree.fight.state(), tree.views);
    }

    Json open(const Json& requests) {
        struct Staged {
            std::optional<Tree> tree;
            int steps = 0;
            std::uint64_t observation = 0;
            Json view;
            bool include_graph = false;
        };
        std::vector<Staged> staged;
        staged.reserve(requests.size());
        std::vector<Tree*> active;
        for (const auto& request : requests) {
            auto fight = replay(request.at("start"), request.at("moves"));
            const auto obs = combat::observation_key(fight.state());
            auto view = combat::view(fight.state());
            const bool include_graph = request.value("include_graph", false);
            if (fight.over()) {
                staged.push_back({std::nullopt, fight.steps(), obs, std::move(view), include_graph});
                continue;
            }
            search::SearchConfig config = request.contains("search")
                ? search_config(request.at("search"))
                : search::SearchConfig{0, 1, 0, 0, search::Gumbel::Settings{}.considered_actions,
                                       request.value("draw", std::uint64_t{0}), true};
            config.validate();
            auto random = fight.decision_random();
            auto created = config.create(fight.state(), config.sims, random);
            Tree tree{std::move(fight), std::move(created.search), &created.gumbel, {}, 0};
            tree.views.emplace(obs, view);
            staged.push_back({std::move(tree), 0, obs, std::move(view), include_graph});
        }
        for (auto& s : staged)
            if (s.tree) active.push_back(&*s.tree);
        answer(active);

        Json out = Json::array();
        for (auto& s : staged) {
            if (!s.tree) {
                out.push_back({{"id", nullptr}, {"steps", s.steps}, {"over", true},
                               {"observation", std::to_string(s.observation)}, {"view", std::move(s.view)}});
                continue;
            }
            const int id = next_id++;
            auto [it, _] = trees.emplace(id, std::move(*s.tree));
            auto& tree = it->second;
            Json moves = Json::array();
            for (const auto action : legal(tree)) moves.push_back(move_json(action, tree.fight.state()));
            const auto graph = tree.gumbel->graph();
            Json entry{{"id", id}, {"steps", tree.fight.steps()}, {"over", false},
                       {"observation", std::to_string(s.observation)}, {"view", std::move(s.view)},
                       {"moves", std::move(moves)}, {"root", root_json(graph, tree.requests)},
                       {"decision", decision_json(tree.search->result())}, {"network_requests", tree.requests}};
            if (s.include_graph) entry["graph"] = tree_graph(tree, graph);
            out.push_back(std::move(entry));
        }
        return {{"trees", std::move(out)}};
    }

    Json simulate(const Json& command) {
        const auto& picks = command.at("picks");
        const bool include_graph = command.value("include_graph", false);
        struct Batch { std::string key; Tree* tree; std::vector<search::Gumbel::Pick> steer; };
        std::vector<Batch> batches;
        for (const auto& [id, wave] : picks.items()) {
            auto& tree = at(Json(std::stoi(id)));
            std::vector<search::Gumbel::Pick> steer;
            for (const auto& pick : wave) {
                const auto& head = pick.at(0);
                const int world = pick.at(1).get<int>();
                if (head.is_array()) {
                    const bool descend = pick.size() >= 3 ? pick.at(2).get<bool>() : false;
                    steer.push_back({head.get<std::vector<std::size_t>>(), world, descend});
                } else {
                    const bool descend = pick.size() >= 3 ? pick.at(2).get<bool>() : true;
                    steer.push_back({{head.get<std::size_t>()}, world, descend});
                }
            }
            batches.push_back({id, &tree, std::move(steer)});
        }
        std::vector<Tree*> active;
        active.reserve(batches.size());
        for (const auto& b : batches) {
            b.tree->gumbel->simulate(b.steer);
            active.push_back(b.tree);
        }
        answer(active);
        Json roots = Json::object();
        Json graphs = Json::object();
        Json decisions = Json::object();
        for (const auto& b : batches) {
            const auto graph = b.tree->gumbel->graph();
            roots[b.key] = root_json(graph, b.tree->requests);
            if (include_graph) {
                graphs[b.key] = tree_graph(*b.tree, graph);
                decisions[b.key] = decision_json(b.tree->search->result());
            }
        }
        Json response{{"roots", std::move(roots)}};
        if (include_graph) {
            response["graphs"] = std::move(graphs);
            response["decisions"] = std::move(decisions);
        }
        return response;
    }

    Json state(const Json& command) {
        const auto& tree = at(command.at("tree"));
        const auto path = command.at("path").get<std::vector<std::size_t>>();
        const int world = command.at("world").get<int>();
        const auto s = tree.gumbel->state(path, world);
        const auto obs = combat::observation_key(s);
        Json moves = Json::array();
        if (s.outcome == sts::Outcome::UNDECIDED)
            for (const auto action : combat::legal_actions(s)) moves.push_back(move_json(action, s));
        Json response{{"view", combat::view(s)}, {"moves", std::move(moves)}, {"observation", std::to_string(obs)}};
        if (command.contains("worlds")) {
            const int count = command.at("worlds").get<int>();
            if (count < 0) throw std::invalid_argument{"state.worlds must be nonnegative"};
            Json reaching = Json::array();
            for (int w = 0; w < count; ++w) {
                try {
                    if (combat::observation_key(tree.gumbel->state(path, w)) == obs) reaching.push_back(w);
                } catch (const std::invalid_argument&) {
                }
            }
            response["reaching"] = std::move(reaching);
        }
        return response;
    }

    Json run(const Json& command) {
        const auto op = command.at("op").get<std::string>();
        if (op == "open") return open(command.at("trees"));
        if (op == "simulate") return simulate(command);
        if (op == "graph") {
            auto& tree = at(command.at("tree"));
            const auto obs = combat::observation_key(tree.fight.state());
            const auto graph = tree.gumbel->graph();
            return {{"decision", decision_json(tree.search->result())},
                    {"network_requests", tree.requests},
                    {"view", tree.views.at(obs)},
                    {"root", root_json(graph, tree.requests)},
                    {"graph", tree_graph(tree, graph)}};
        }
        if (op == "state") return state(command);
        if (op == "close") {
            for (const auto& id : command.at("trees")) trees.erase(id.get<int>());
            return Json::object();
        }
        throw std::invalid_argument{"unknown op " + op};
    }
};

Session::Session(nn::Encoding encoding, Predict predict) : impl_{std::make_unique<Impl>(encoding, std::move(predict))} {}
Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;
Json Session::run(const Json& command) { return impl_->run(command); }
}  // namespace stsrl::inspection
