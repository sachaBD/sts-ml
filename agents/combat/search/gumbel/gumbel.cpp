#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <syncstream>
#include <unordered_map>

namespace stsrl::search {
namespace {
// Values live only on nodes; an edge only counts where it led. Every other value is derived from these.
struct Node {
    double value = 0;  // the network's estimate, or a finished fight's outcome
    double value_sum = 0;  // over every simulation through this node, by any path
    int visits = 0;
    std::vector<std::uint64_t> keys;  // edges, by action key
    std::vector<double> logits, prior;
    std::vector<std::vector<std::pair<Node*, int>>> reached;  // per edge: each node it led to, and how often
};

double q(const Node& n) { return n.value_sum / n.visits; }

int visits(const Node& n, std::size_t edge) {
    int count = 0;
    for (const auto& [child, times] : n.reached[edge]) count += times;
    return count;
}

double q(const Node& n, std::size_t edge) {
    double sum = 0;
    for (const auto& [child, times] : n.reached[edge]) sum += times * q(*child);
    return sum / visits(n, edge);
}

int total(const Node& n) {
    int count = 0;
    for (std::size_t e = 0; e < n.keys.size(); ++e) count += visits(n, e);
    return count;
}

// How the search meets hidden information and chance: where a simulation starts, and which node a state is.
// Particles: every simulation plays out in one sampled world, and nodes are told apart by what the player
// can see, so all worlds and all move orders share one graph.
class Particles {
public:
    Particles(const sts::BattleContext& observed, int count) : worlds_{combat::particles(observed, count)} {}
    sts::BattleContext start(std::mt19937_64& rng) const {
        return worlds_[std::uniform_int_distribution<std::size_t>{0, worlds_.size() - 1}(rng)];
    }
    static std::uint64_t node_key(const sts::BattleContext& state) { return combat::observation_key(state); }

private:
    std::vector<sts::BattleContext> worlds_;
};
using Chance = Particles;

std::vector<double> softmax(std::vector<double> x) {
    const double top = *std::max_element(x.begin(), x.end());
    double sum = 0;
    for (auto& v : x) sum += v = std::exp(v - top);
    for (auto& v : x) v /= sum;
    return x;
}

// mctx qtransform_completed_by_mix_value: completed Q (unvisited = mixed value), shifted to zero, scaled (no min-max rescale).
std::vector<double> sigma(const Node& n, const Gumbel::Settings& s) {
    double prior_visited = 0, weighted = 0;
    int most = 0;
    for (std::size_t i = 0; i < n.keys.size(); ++i)
        if (const int v = visits(n, i)) {
            prior_visited += n.prior[i]; weighted += n.prior[i] * q(n, i);
            most = std::max(most, v);
        }
    const int sum = total(n);
    const double mixed = prior_visited > 0 ? (n.value + sum * weighted / prior_visited) / (sum + 1) : n.value;
    std::vector<double> completed(n.keys.size());
    for (std::size_t i = 0; i < completed.size(); ++i) completed[i] = visits(n, i) ? q(n, i) : mixed;
    const double low = *std::min_element(completed.begin(), completed.end());
    for (auto& v : completed) v = (s.maxvisit_init + most) * s.value_scale * (v - low);
    return completed;
}

std::vector<double> improved_policy(const Node& n, const Gumbel::Settings& s) {
    auto z = sigma(n, s);
    for (std::size_t i = 0; i < z.size(); ++i) z[i] += n.logits[i];
    return softmax(z);
}

// mctx seq_halving.get_sequence_of_considered_visits, stopping before a partial pass so every round
// visits its considered actions equally.
std::vector<int> halving_schedule(int m, int simulations) {
    if (simulations <= 0) return {};
    m = std::min(m, std::max(1, simulations / 2));
    std::vector<int> out;
    if (m <= 1) { for (int i = 0; i < simulations; ++i) out.push_back(i); return out; }
    const int log2m = int(std::ceil(std::log2(double(m))));
    std::vector<int> visits(m, 0);
    for (int considered = m; int(out.size()) + considered <= simulations; considered = std::max(2, considered / 2)) {
        const int extra = std::max(1, int(double(simulations) / (log2m * considered)));
        for (int r = 0; r < extra && int(out.size()) + considered <= simulations; ++r) {
            for (int i = 0; i < considered; ++i) out.push_back(visits[i]);
            for (int i = 0; i < considered; ++i) ++visits[i];
        }
    }
    return out;
}

// Edges are public keys because card slots can differ between worlds.
void play(sts::BattleContext& state, std::uint64_t key) {
    for (const auto a : combat::legal_actions(state))
        if (combat::action_key(state, a) == key) { a.execute(state); return; }
    throw std::runtime_error{"gumbel: action missing in a particle"};
}
}  // namespace

struct Gumbel::Impl {
    struct Step { Node* node; std::size_t edge; Node* reached; };
    using Path = std::vector<Step>;
    struct Leaf { Node* node; sts::BattleContext state; std::vector<Action> actions; };

    Settings s;
    std::uint64_t fight_seed;
    std::mt19937_64 rng;
    Chance chance;
    std::vector<Action> actions;
    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> nodes;  // by Chance::node_key
    Node& root;
    std::vector<double> gumbel;
    std::vector<int> schedule;
    int simulation = 0;

    // The wave: every simulation of one Sequential Halving pass, descended from the same statistics.
    // Nothing is backed up until the whole wave is answered, so answers cannot change which leaves it asks for.
    std::vector<Leaf> leaves;
    std::vector<Query> queries;  // views of leaves
    std::vector<std::pair<Path, Node*>> backups;  // in issue order

    Impl(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
        : s{settings}, fight_seed{observed.seed}, rng{seed}, chance{observed, settings.particles}, actions{combat::legal_actions(observed)},
          root{*(nodes[Chance::node_key(observed)] = std::make_unique<Node>())},
          leaves{{&root, observed, actions}}, queries{{leaves[0].state, leaves[0].actions}}, backups{{{}, &root}} {}

    void evaluated(std::span<const Evaluation> e) {
        if (leaves.empty() || e.size() != leaves.size()) throw std::invalid_argument{"gumbel: one evaluation per pending query"};
        for (std::size_t i = 0; i < e.size(); ++i) expand(leaves[i], e[i]);
        if (leaves.front().node == &root) {
            std::extreme_value_distribution<double> g;
            for (std::size_t i = 0; i < actions.size(); ++i) gumbel.push_back(s.gumbel_scale * g(rng));
            schedule = halving_schedule(std::min<int>(s.considered_actions, int(actions.size())), s.simulations);
        }
        leaves.clear();
        queries.clear();
        do {
            for (const auto& [path, end] : backups) backup(path, *end);
            backups.clear();
            if (simulation < int(schedule.size()))
                for (const auto edge : ranked(schedule[simulation], pass_width())) simulate(edge);
        } while (leaves.empty() && !backups.empty());
        for (const auto& leaf : leaves) queries.push_back({leaf.state, leaf.actions});
    }

    // The schedule repeats a visit count once per action in the pass.
    std::size_t pass_width() const {
        const auto first = schedule.begin() + simulation;
        return std::find_if(first, schedule.end(), [&](int v) { return v != *first; }) - first;
    }

    // Descends until a finished fight or a node already on the path (backed up with the wave), or a node
    // without an evaluation (left for the evaluator, once however many simulations reach it).
    void simulate(std::size_t root_edge) {
        auto state = chance.start(rng);
        Path path;
        Node* node = &root;
        ++simulation;
        for (;;) {
            const auto edge = node == &root ? root_edge : interior_choice(*node);
            play(state, node->keys[edge]);
            auto& next = nodes[Chance::node_key(state)];
            const bool created = !next;
            if (created) next = std::make_unique<Node>();
            const bool cycle = next.get() == node ||
                std::ranges::any_of(path, [&](const Step& step) { return step.node == next.get(); });
            path.push_back({node, edge, next.get()});
            if (cycle) {
                std::osyncstream error{std::cerr};
                error << "ERROR gumbel: cycle truncated; fight_seed=" << fight_seed
                      << " simulation=" << simulation << " observation=" << Chance::node_key(state)
                      << " bootstrap=" << next->value << " path_action_keys=[";
                for (const auto& step : path) error << step.node->keys[step.edge] << ',';
                error << "]\n" << state;
                backups.emplace_back(std::move(path), next.get());
                return;
            }
            if (state.outcome != sts::Outcome::UNDECIDED) {
                next->value = state.outcome == sts::Outcome::PLAYER_VICTORY;
                backups.emplace_back(std::move(path), next.get());
                return;
            }
            if (created) leaves.push_back({next.get(), state, combat::legal_actions(state)});
            if (std::ranges::find(leaves, next.get(), &Leaf::node) != leaves.end()) {
                backups.emplace_back(std::move(path), next.get());
                return;
            }
            node = next.get();
        }
    }

    // The one place values change: `end`'s value is counted at every node on the path, and each edge
    // counts the node it reached.
    static void backup(const Path& path, Node& end) {
        const double value = end.value;
        for (const auto& [node, edge, reached] : path) {
            auto& counts = node->reached[edge];
            const auto found = std::ranges::find(counts, reached, &std::pair<Node*, int>::first);
            if (found == counts.end()) counts.emplace_back(reached, 1); else ++found->second;
            node->value_sum += value; ++node->visits;
        }
        // A cycle's endpoint was already counted as a source on this path.
        if (std::ranges::none_of(path, [&](const Step& step) { return step.node == &end; })) {
            end.value_sum += value; ++end.visits;
        }
    }

    static void expand(const Leaf& leaf, const Evaluation& e) {
        auto& n = *leaf.node;
        n.value = e.value;
        n.logits.assign(e.logits.begin(), e.logits.end());
        n.prior = softmax(n.logits);
        for (const auto a : leaf.actions) n.keys.push_back(combat::action_key(leaf.state, a));
        n.reached.resize(n.keys.size());
    }

    // Root: the best `count` by gumbel + logits + sigma among actions with exactly `considered` visits.
    std::vector<std::size_t> ranked(int considered, std::size_t count) const {
        const auto q = sigma(root, s);
        std::vector<std::size_t> edges;
        for (std::size_t i = 0; i < q.size(); ++i)
            if (visits(root, i) == considered) edges.push_back(i);
        std::ranges::stable_sort(edges, std::greater{}, [&](std::size_t i) { return gumbel[i] + root.logits[i] + q[i]; });
        edges.resize(std::min(count, edges.size()));
        return edges;
    }

    // Interior: argmax of improved policy - visits / (1 + total).
    std::size_t interior_choice(const Node& n) const {
        const auto pi = improved_policy(n, s);
        const int sum = total(n);
        std::size_t best = 0; double best_score = -INFINITY;
        for (std::size_t i = 0; i < pi.size(); ++i) {
            const double score = pi[i] - double(visits(n, i)) / (1 + sum);
            if (score > best_score) best = i, best_score = score;
        }
        return best;
    }

    Graph graph() const {
        if (!leaves.empty()) throw std::logic_error{"gumbel: graph requires a completed search"};
        std::unordered_map<const Node*, std::uint64_t> observations;
        for (const auto& [key, node] : nodes) observations.emplace(node.get(), key);
        Graph out{observations.at(&root), {}, gumbel};
        for (const auto& [key, node] : nodes) {
            const auto& n = *node;
            Graph::Node snapshot{key, n.keys.empty(), n.value, q(n), n.visits, {}};
            // Every nonterminal node has been evaluated when the search is complete.
            if (!snapshot.terminal) {
                const auto scores = sigma(n, s), policy = improved_policy(n, s);
                for (std::size_t i = 0; i < n.keys.size(); ++i) {
                    const int count = visits(n, i);
                    Graph::Edge edge{n.keys[i], n.logits[i], n.prior[i], scores[i], policy[i],
                                     count ? std::optional{q(n, i)} : std::nullopt, count, {}};
                    for (const auto& [child, times] : n.reached[i])
                        edge.children.push_back({observations.at(child), times});
                    snapshot.edges.push_back(std::move(edge));
                }
            }
            out.nodes.push_back(std::move(snapshot));
        }
        std::ranges::sort(out.nodes, {}, &Graph::Node::observation);
        return out;
    }

    Result result() const {
        int most = 0, sum = 0;
        double value_sum = 0;
        for (std::size_t i = 0; i < root.keys.size(); ++i)
            if (const int v = visits(root, i)) { most = std::max(most, v); sum += v; value_sum += v * q(root, i); }
        Result r{actions[ranked(most, 1).front()], actions, {}, float(sum ? value_sum / sum : root.value)};
        for (double p : improved_policy(root, s)) r.policy.push_back(float(p));
        return r;
    }
};

Gumbel::Gumbel(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
    : impl_{std::make_unique<Impl>(observed, settings, seed)} {}
Gumbel::~Gumbel() = default;

std::span<const Query> Gumbel::pending() const { return impl_->queries; }
void Gumbel::evaluated(std::span<const Evaluation> evaluations) { impl_->evaluated(evaluations); }
Gumbel::Graph Gumbel::graph() const { return impl_->graph(); }
Result Gumbel::result() const { return impl_->result(); }
}  // namespace stsrl::search
