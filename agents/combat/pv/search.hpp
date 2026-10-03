#pragma once

#include "agents/combat/pv/evaluator.hpp"
#include "agents/combat/pv/objective.hpp"
#include "agents/combat/pv/selection.hpp"

#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>

namespace stsrl::pv {

struct SearchSettings {
    int batch_size = 32;
    int maximum_actions = 512;
    double noise_alpha = 0.3;
    double noise_fraction = 0;  // evaluation: off; self-play can explicitly set 0.25
};

// The old search class is used only for public-observation/action utilities, never as a search driver.
std::vector<sts::search::Action> legal_actions(const sts::BattleContext&);
std::uint64_t observation_key(const sts::BattleContext&);
std::uint64_t action_key(const sts::BattleContext&, sts::search::Action);
std::vector<double> policy_priors(const Prediction&, std::size_t action_count);
void validate_root_state(const sts::BattleContext&);
Prediction evaluate_root(Evaluator&, const sts::BattleContext&);
void add_root_noise(std::vector<double>& priors, const SearchSettings&, std::mt19937_64&);

// Dedicated PV tree. Construction requires a root prediction; every traversable node has evaluated priors.
// Score and Selection are independent compile-time policies. Default: AlphaZero-style PUCT + argmax.
template<class Score = PuctScore, class Selection = ArgmaxSelection>
class Search {
public:
    struct Node;
    struct Edge {
        sts::search::Action action;
        std::uint64_t key;
        double prior = 0;
        std::int64_t visits = 0;
        double value_sum = 0;
        int in_flight = 0;
        std::map<std::uint64_t, std::unique_ptr<Node>> outcomes;
    };

    struct Node {
        std::vector<Edge> edges;
        std::optional<double> value;  // absent until this node's network evaluation completes
        std::int64_t visits = 0;
        int in_flight = 0;
    };

    Search(const sts::BattleContext& observed, std::vector<sts::BattleContext> particles,
           const Prediction& root_prediction, SearchSettings settings = {}, Score score = {}, Selection select = {})
        : particles_{std::move(particles)}, settings_{settings}, score_{std::move(score)},
          select_{std::move(select)}, random_{observation_key(observed)}, root_{make_root(observed)} {
        if (particles_.empty() || settings_.batch_size < 1 || settings_.maximum_actions < 1) {
            throw std::invalid_argument{"PV: invalid particles or search bounds"};
        }
        for (const auto& particle : particles_) {
            if (observation_key(particle) != observation_key(observed)) {
                throw std::invalid_argument{"PV: particle does not match the root observation"};
            }
            (void)ordered_actions(root_, particle);
        }

        evaluate_node(root_, root_prediction);
        std::vector<double> priors;
        for (const auto& edge : root_.edges) priors.push_back(edge.prior);
        add_root_noise(priors, settings_, random_);
        for (std::size_t i = 0; i < priors.size(); ++i) root_.edges[i].prior = priors[i];
    }

    const Node& root() const { return root_; }
    std::int64_t simulations() const { return simulations_; }

    sts::search::Action selected_action() const {
        if (root_.in_flight || simulations_ == 0) throw std::logic_error{"PV: search is incomplete"};
        const auto best = std::max_element(root_.edges.begin(), root_.edges.end(),
            [](const Edge& a, const Edge& b) { return a.visits < b.visits; });
        return best->action;  // played move: most visits; separate from traversal selection
    }

    // Evaluate is a batched callable: span<const Inputs> -> vector<Prediction>.
    // Keeping the boundary generic also lets algorithm checks use known predictions, not an imported runtime.
    template<class Evaluate>
    void run(Evaluate&& evaluate, std::int64_t additional_simulations) {
        if (additional_simulations < 1 || additional_simulations > std::numeric_limits<std::int64_t>::max() - simulations_ ||
            root_.in_flight) {
            throw std::invalid_argument{"PV: invalid simulation budget or unfinished batch"};
        }
        const auto budget = simulations_ + additional_simulations;

        while (simulations_ < budget) {
            std::vector<Request> requests;
            int pending_paths = 0;

            // Reserve paths using virtual loss. Multiple paths to one pending node share one evaluation.
            for (int slot = 0; slot < settings_.batch_size && simulations_ + pending_paths < budget; ++slot) {
                auto state = particles_[std::uniform_int_distribution<std::size_t>{0, particles_.size() - 1}(random_)];
                Node* node = &root_;
                Path path;

                for (int depth = 0; ; ++depth) {
                    const auto moves = ordered_actions(*node, state);

                    if (!node->value) {
                        auto request = std::find_if(requests.begin(), requests.end(),
                            [node](const Request& r) { return r.node == node; });
                        if (request == requests.end()) {
                            requests.push_back({node, std::move(state), {std::move(path)}});
                        } else {
                            request->paths.push_back(std::move(path));
                        }
                        ++pending_paths;
                        break;  // never traverse an unevaluated node with guessed/uniform priors
                    }

                    if (depth == settings_.maximum_actions) {
                        backup(path, *node->value);  // explicit depth cutoff, using this node's evaluated V
                        break;
                    }

                    const auto choice = choose(*node);
                    auto& edge = node->edges.at(choice);
                    const auto action = moves.at(choice);
                    ++node->in_flight;
                    ++edge.in_flight;
                    path.emplace_back(node, choice);

                    action.execute(state);
                    if (state.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE) {
                        throw std::runtime_error{"PV: unsupported simulator effect"};
                    }
                    if (state.outcome != sts::Outcome::UNDECIDED) {
                        backup(path, CombatObjective::terminal_value(state));
                        break;
                    }

                    auto& child = edge.outcomes[observation_key(state)];
                    if (!child) child = std::make_unique<Node>(make_node(state));
                    node = child.get();
                }
            }

            // One network call for the distinct leaves, followed by value backups for all reserved paths.
            if (!requests.empty()) {
                std::vector<Inputs> batch;
                for (const auto& request : requests) {
                    batch.push_back(encode(request.state, ordered_actions(*request.node, request.state)));
                }

                const auto predictions = evaluate(std::span<const Inputs>{batch});
                if (predictions.size() != requests.size()) throw std::runtime_error{"PV: prediction count mismatch"};
                for (std::size_t i = 0; i < requests.size(); ++i) {
                    evaluate_node(*requests[i].node, predictions[i]);
                    for (auto& path : requests[i].paths) backup(path, predictions[i].value);
                }
            }
        }
    }

private:
    using Path = std::vector<std::pair<Node*, std::size_t>>;
    struct Request { Node* node; sts::BattleContext state; std::vector<Path> paths; };

    static Node make_root(const sts::BattleContext& state) {
        validate_root_state(state);
        return make_node(state);
    }

    static Node make_node(const sts::BattleContext& state) {
        Node node;
        for (const auto action : legal_actions(state)) {
            node.edges.push_back(Edge{.action = action, .key = action_key(state, action), .outcomes = {}});
        }
        return node;
    }

    // Action indices can differ between particles (e.g. Secret Technique's draw-pile menu).
    // Match the entire legal menu by public semantic key, and return actions in the node's prior order.
    static std::vector<sts::search::Action> ordered_actions(const Node& node, const sts::BattleContext& state) {
        const auto available = legal_actions(state);
        if (available.size() != node.edges.size()) {
            throw std::runtime_error{"PV: incompatible legal menus at a shared public history"};
        }
        std::unordered_map<std::uint64_t, sts::search::Action> by_key;
        for (const auto action : available) by_key.emplace(action_key(state, action), action);

        std::vector<sts::search::Action> ordered;
        for (const auto& edge : node.edges) {
            const auto found = by_key.find(edge.key);
            if (found == by_key.end()) throw std::runtime_error{"PV: incompatible action at a shared public history"};
            ordered.push_back(found->second);
        }
        return ordered;
    }

    static void evaluate_node(Node& node, const Prediction& prediction) {
        if (node.value) throw std::logic_error{"PV: node evaluated twice"};
        const auto priors = policy_priors(prediction, node.edges.size());
        for (std::size_t i = 0; i < priors.size(); ++i) node.edges[i].prior = priors[i];
        node.value = prediction.value;
    }

    std::size_t choose(const Node& node) {
        if (!node.value) throw std::logic_error{"PV: cannot traverse an unevaluated node"};
        std::vector<double> scores;
        for (const auto& edge : node.edges) {
            const auto visits = edge.visits + edge.in_flight;
            const double q = visits ? edge.value_sum / visits : 0;
            const double score = score_(q, edge.prior, node.visits + node.in_flight, visits);
            if (!std::isfinite(score)) throw std::runtime_error{"PV: non-finite selection score"};
            scores.push_back(score);
        }
        const auto chosen = select_(std::span<const double>{scores}, random_);
        if (chosen >= node.edges.size()) throw std::logic_error{"PV: selection policy returned an invalid edge"};
        return chosen;
    }

    void backup(Path& path, double value) {
        validate_value(value);
        for (auto [node, index] : path) {
            auto& edge = node->edges.at(index);
            if (node->in_flight < 1 || edge.in_flight < 1) throw std::logic_error{"PV: unreserved backup path"};
            --node->in_flight;
            --edge.in_flight;
            ++node->visits;
            ++edge.visits;
            edge.value_sum += value;
        }
        ++simulations_;
    }

    std::vector<sts::BattleContext> particles_;
    SearchSettings settings_;
    Score score_;
    Selection select_;
    std::mt19937_64 random_;
    Node root_;
    std::int64_t simulations_ = 0;
};

}  // namespace stsrl::pv
