#include "agents/mcts_agent.hpp"

#include "combat/environment.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stsrl {

namespace {

struct Node;

struct Edge {
    explicit Edge(SearchActionKey key) : key{key} {}

    SearchActionKey key;
    std::size_t availability{};
    std::size_t visits{};
    double value_sum{};
    std::unique_ptr<Node> child;
};

struct Node {
    std::vector<Edge> edges;
};

struct Candidate {
    Edge* edge;
    std::size_t action_index;
};

}  // namespace

struct MctsAgent::Impl {
    Impl(const std::uint64_t seed, MctsConfig config)
        : config{config}, random{seed} {
        if (config.simulations == 0) {
            throw std::invalid_argument{"MCTS requires at least one simulation"};
        }
    }

    [[nodiscard]] double rollout(CombatEnvironment environment) {
        for (std::size_t depth = 0;
             !environment.done() && depth < config.rollout_limit;
             ++depth) {
            const auto actions = environment.search_actions();
            std::uniform_int_distribution<std::size_t> choice{0, actions.size() - 1};
            environment.step(actions[choice(random)].index);
        }
        return environment.combat_value();
    }

    [[nodiscard]] double search(CombatEnvironment& environment, Node& node) {
        if (environment.done()) {
            return environment.combat_value();
        }

        const auto actions = environment.search_actions();
        std::vector<Candidate> candidates;
        candidates.reserve(actions.size());
        node.edges.reserve(node.edges.size() + actions.size());

        for (const auto& action : actions) {
            auto edge = std::find_if(node.edges.begin(), node.edges.end(), [&action](const Edge& candidate) { return candidate.key == action.key; });
            if (edge == node.edges.end()) edge = node.edges.emplace(node.edges.end(), action.key);
            ++edge->availability;
            candidates.push_back({&*edge, action.index});
        }

        if (candidates.empty()) {
            throw std::logic_error{"non-terminal combat has no legal actions"};
        }

        std::vector<Candidate> unexplored;
        std::ranges::copy_if(
            candidates, std::back_inserter(unexplored),
            [](const Candidate& candidate) { return candidate.edge->visits == 0; });

        Candidate selected{};
        bool expanded = false;
        if (!unexplored.empty()) {
            std::uniform_int_distribution<std::size_t> choice{0, unexplored.size() - 1};
            selected = unexplored[choice(random)];
            expanded = true;
        } else {
            const auto score = [this](const Candidate& candidate) {
                const auto& edge = *candidate.edge;
                const auto mean = edge.value_sum / static_cast<double>(edge.visits);
                const auto explore = config.exploration * std::sqrt(
                    std::log(static_cast<double>(edge.availability))
                    / static_cast<double>(edge.visits));
                return mean + explore;
            };
            selected = *std::max_element(
                candidates.begin(), candidates.end(),
                [&score](const Candidate& lhs, const Candidate& rhs) {
                    return score(lhs) < score(rhs);
                });
        }

        environment.step(selected.action_index);
        auto& edge = *selected.edge;
        double value{};
        if (expanded) {
            edge.child = std::make_unique<Node>();
            value = rollout(std::move(environment));
        } else {
            value = search(environment, *edge.child);
        }

        ++edge.visits;
        edge.value_sum += value;
        return value;
    }

    [[nodiscard]] MctsResult search_result(CombatEnvironment& environment) {
        if (environment.done()) {
            throw std::logic_error{"MCTS cannot act in a finished combat"};
        }

        Node root;
        for (std::size_t simulation = 0; simulation < config.simulations; ++simulation) {
            auto determinization = environment.determinized(random());
            [[maybe_unused]] const auto value = search(determinization, root);
        }

        const auto actions = environment.search_actions();
        const Edge* best_edge = nullptr;
        std::size_t best_action{};
        for (const auto& action : actions) {
            const auto edge = std::find_if(root.edges.begin(), root.edges.end(), [&action](const Edge& candidate) { return candidate.key == action.key; });
            if (edge != root.edges.end()
                && (best_edge == nullptr || edge->visits > best_edge->visits)) {
                best_edge = &*edge;
                best_action = action.index;
            }
        }

        if (best_edge == nullptr) throw std::logic_error{"MCTS produced no legal root action"};
        MctsResult result{.chosen_action = best_action};
        std::vector<SearchActionKey> seen;
        for (const auto& action : actions) {
            if (std::find(seen.begin(), seen.end(), action.key) != seen.end()) continue;
            seen.push_back(action.key);
            const auto edge = std::find_if(root.edges.begin(), root.edges.end(), [&action](const Edge& x) { return x.key == action.key; });
            if (edge != root.edges.end()) {
                result.root_visits += edge->visits;
                result.root_value += edge->value_sum;
                result.actions.push_back({action.index, edge->visits, edge->visits ? edge->value_sum / edge->visits : 0.0});
            }
        }
        result.root_value = result.root_visits ? result.root_value / result.root_visits : 0.0;
        return result;
    }

    MctsConfig config;
    std::mt19937_64 random;
};

MctsAgent::MctsAgent(const std::uint64_t seed, MctsConfig config)
    : impl_{std::make_unique<Impl>(seed, config)} {}

MctsAgent::~MctsAgent() = default;
MctsAgent::MctsAgent(MctsAgent&&) noexcept = default;
MctsAgent& MctsAgent::operator=(MctsAgent&&) noexcept = default;

MctsResult MctsAgent::search(CombatEnvironment& environment) { return impl_->search_result(environment); }

std::size_t MctsAgent::choose_action(CombatEnvironment& environment) { return search(environment).chosen_action; }

}  // namespace stsrl
