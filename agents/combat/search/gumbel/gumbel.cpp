#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <syncstream>
#include <unordered_map>
#include <unordered_set>

namespace stsrl::search {
namespace {
// Values live on nodes; an edge's value is derived from the nodes it led to.
struct Node {
    struct Outcome {
        Node* node;
        int visits;
        bool bootstrap;  // closes a cycle, so counts the successor's network estimate
    };
    double value = 0;  // the network's estimate, or a finished fight's outcome
    double backed_value = 0;  // Backup's value of this node
    std::vector<Node*> parents;  // sources of non-bootstrap outcomes, which read backed_value
    std::vector<std::uint64_t> keys;  // edges, by action key
    std::vector<double> logits, prior;
    std::vector<std::vector<Outcome>> reached;  // per edge: each node it led to
    std::vector<int> waiting;  // per edge: simulations of the current wave, not yet backed up
    std::vector<Gumbel::TransitionKind> transitions;  // chance is sticky across particles and paths
};

int visits(const Node& n, std::size_t edge) {
    int count = 0;
    for (const auto& child : n.reached[edge]) count += child.visits;
    return count;
}

double link_value(const Node::Outcome& outcome) {
    return outcome.bootstrap ? outcome.node->value : outcome.node->backed_value;
}

// The expectation over sampled outcomes.
double q(const Node& n, std::size_t edge) {
    double sum = 0;
    for (const auto& outcome : n.reached[edge]) sum += outcome.visits * link_value(outcome);
    return sum / visits(n, edge);
}

struct ActionValue { int visits; double q; };

// Values a choice node as its best visited action, or its network estimate before any visit.
struct HardMax {
    static double value(double network, std::span<const ActionValue> visited) {
        if (visited.empty()) return network;
        return std::ranges::max(visited, {}, &ActionValue::q).q;
    }
};

using Backup = HardMax;  // the backup rule the search uses

// Returns whether the value changed.
bool revalue(Node& n) {
    std::vector<ActionValue> visited;
    for (std::size_t edge = 0; edge < n.keys.size(); ++edge)
        if (const int count = visits(n, edge)) visited.push_back({count, q(n, edge)});
    const double value = Backup::value(n.value, visited);
    const bool changed = value != n.backed_value;
    n.backed_value = value;
    return changed;
}

// Whether `to` is reachable from `from` along non-bootstrap outcomes.
bool reaches(const Node& from, const Node* to) {
    std::vector<const Node*> pending{&from};
    std::unordered_set<const Node*> seen;
    while (!pending.empty()) {
        const auto* node = pending.back();
        pending.pop_back();
        if (node == to) return true;
        if (!seen.insert(node).second) continue;
        for (const auto& outcomes : node->reached)
            for (const auto& outcome : outcomes)
                if (!outcome.bootstrap) pending.push_back(outcome.node);
    }
    return false;
}

int total(const Node& n) {
    int count = 0;
    for (std::size_t e = 0; e < n.keys.size(); ++e) count += visits(n, e);
    return count;
}

class Particles {
public:
    Particles(const sts::BattleContext& observed, int count, std::uint64_t stream)
        : worlds_{combat::particles(observed, count, stream)} {}
    sts::BattleContext start(std::mt19937_64& rng) const {
        return worlds_[std::uniform_int_distribution<std::size_t>{0, worlds_.size() - 1}(rng)];
    }

private:
    std::vector<sts::BattleContext> worlds_;
};

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

// Which action a simulation takes at an interior node, and the policy target a node reports.
struct ImprovedPolicy {
    static std::vector<double> policy(const Node& n, const Gumbel::Settings& s) {
        auto z = sigma(n, s);
        for (std::size_t i = 0; i < z.size(); ++i) z[i] += n.logits[i];
        return softmax(z);
    }

    // Argmax of policy - visits / (1 + total), counting the wave's simulations as visits so that simulations
    // sharing a wave spread out.
    static std::size_t choose(const Node& n, const Gumbel::Settings& s) {
        const auto pi = policy(n, s);
        const int sum = total(n) + std::accumulate(n.waiting.begin(), n.waiting.end(), 0);
        std::size_t best = 0; double best_score = -INFINITY;
        for (std::size_t i = 0; i < pi.size(); ++i) {
            const double score = pi[i] - double(visits(n, i) + n.waiting[i]) / (1 + sum);
            if (score > best_score) best = i, best_score = score;
        }
        return best;
    }
};

using Selection = ImprovedPolicy;  // the selection rule the search uses

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
sts::search::TransitionEffects play(sts::BattleContext& state, std::uint64_t key) {
    for (const auto a : combat::legal_actions(state))
        if (combat::action_key(state, a) == key) return a.execute(state);
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
    Particles particles;
    std::vector<Action> actions;
    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> nodes;  // by public observation
    Node& root;
    std::vector<double> gumbel;
    std::vector<int> schedule;
    int simulation = 0;

    // One Sequential Halving pass, backed up only once fully answered, so answers cannot steer it.
    std::vector<Leaf> leaves;
    std::vector<Query> queries;
    std::vector<Path> backups;

    Impl(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
        : s{settings}, fight_seed{observed.seed}, rng{seed},
          // Keep sampling stable within a turn to improve answer reuse.
          particles{observed, settings.particles,
                    observed.seed ^ (std::uint64_t(observed.turn) << 32) ^ (settings.world_draw * 0x9E3779B97F4A7C15ull)},
          actions{combat::legal_actions(observed)},
          root{*(nodes[combat::observation_key(observed)] = std::make_unique<Node>())},
          leaves{{&root, observed, actions}}, queries{{leaves[0].state, leaves[0].actions}} {}

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
            for (const auto& path : backups) backup(path);
            backups.clear();
            if (simulation < int(schedule.size())) next_wave();
        } while (leaves.empty() && !backups.empty());
        for (const auto& leaf : leaves) queries.push_back({leaf.state, leaf.actions});
    }

    // Passes of one round, so late rounds with few actions still fill a wave.
    void next_wave() {
        const auto width = pass_width();
        const auto edges = ranked(schedule[simulation], width);
        for (std::size_t size = 0; size + width <= std::max<std::size_t>(width, s.wave); size += width) {
            if (simulation == int(schedule.size()) || pass_width() != width) return;
            for (const auto edge : edges) simulate(edge);
        }
    }

    std::size_t pass_width() const {
        const auto first = schedule.begin() + simulation;
        return std::find_if(first, schedule.end(), [&](int v) { return v != *first; }) - first;
    }

    // Descends until a finished fight, a cycle or an unevaluated node.
    void simulate(std::size_t root_edge) {
        auto state = particles.start(rng);
        Path path;
        Node* node = &root;
        ++simulation;
        for (;;) {
            const auto edge = node == &root ? root_edge : Selection::choose(*node, s);
            const auto effects = play(state, node->keys[edge]);
            auto& transition = node->transitions[edge];
            if (transition != TransitionKind::chance)
                transition = effects.isChance() ? TransitionKind::chance : TransitionKind::deterministic;
            auto& next = nodes[combat::observation_key(state)];
            const bool created = !next;
            if (created) next = std::make_unique<Node>();
            const bool cycle = next.get() == node ||
                std::ranges::any_of(path, [&](const Step& step) { return step.node == next.get(); });
            path.push_back({node, edge, next.get()});
            ++node->waiting[edge];
            if (cycle) {
                std::osyncstream error{std::cerr};
                error << "ERROR gumbel: cycle truncated; fight_seed=" << fight_seed
                      << " simulation=" << simulation << " observation=" << combat::observation_key(state)
                      << " bootstrap=" << next->value << " path_action_keys=[";
                for (const auto& step : path) error << step.node->keys[step.edge] << ',';
                error << "]\n" << state;
                backups.push_back(std::move(path));
                return;
            }
            if (state.outcome != sts::Outcome::UNDECIDED) {
                next->value = state.outcome == sts::Outcome::PLAYER_VICTORY;
                revalue(*next);
                backups.push_back(std::move(path));
                return;
            }
            if (created) leaves.push_back({next.get(), state, combat::legal_actions(state)});
            if (std::ranges::find(leaves, next.get(), &Leaf::node) != leaves.end()) {
                backups.push_back(std::move(path));
                return;
            }
            node = next.get();
        }
    }

    // Each edge counts the node it reached. A new outcome bootstraps if it closes a cycle, so the rest stay
    // acyclic and every changed value can be pushed to its parents until none changes.
    static void backup(const Path& path) {
        std::vector<Node*> stale;
        for (const auto& [node, edge, reached] : path) {
            --node->waiting[edge];
            auto& outcomes = node->reached[edge];
            const auto found = std::ranges::find(outcomes, reached, &Node::Outcome::node);
            if (found != outcomes.end()) {
                ++found->visits;
            } else if (reaches(*reached, node)) {
                outcomes.push_back({reached, 1, true});
            } else {
                outcomes.push_back({reached, 1, false});
                reached->parents.push_back(node);
            }
            stale.push_back(node);
        }
        while (!stale.empty()) {
            auto& node = *stale.back();
            stale.pop_back();
            if (revalue(node)) stale.insert(stale.end(), node.parents.begin(), node.parents.end());
        }
    }

    static void expand(const Leaf& leaf, const Evaluation& e) {
        auto& n = *leaf.node;
        n.value = e.value;
        revalue(n);
        n.logits.assign(e.logits.begin(), e.logits.end());
        n.prior = softmax(n.logits);
        for (const auto a : leaf.actions) n.keys.push_back(combat::action_key(leaf.state, a));
        n.reached.resize(n.keys.size());
        n.waiting.resize(n.keys.size());
        n.transitions.resize(n.keys.size(), TransitionKind::unknown);
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

    Graph graph() const {
        if (!leaves.empty()) throw std::logic_error{"gumbel: graph requires a completed search"};
        std::unordered_map<const Node*, std::uint64_t> observations;
        for (const auto& [key, node] : nodes) observations.emplace(node.get(), key);
        Graph out{observations.at(&root), {}, gumbel};
        for (const auto& [key, node] : nodes) {
            const auto& n = *node;
            Graph::Node snapshot{key, n.keys.empty(), n.value, n.backed_value, {}};
            // Every nonterminal node has been evaluated when the search is complete.
            if (!snapshot.terminal) {
                const auto scores = sigma(n, s), policy = Selection::policy(n, s);
                for (std::size_t i = 0; i < n.keys.size(); ++i) {
                    const int count = visits(n, i);
                    Graph::Edge edge{n.keys[i], n.logits[i], n.prior[i], scores[i], policy[i],
                                     count ? std::optional{q(n, i)} : std::nullopt, count, {}, n.transitions[i]};
                    for (const auto& child : n.reached[i])
                        edge.children.push_back({observations.at(child.node), child.visits, child.bootstrap});
                    snapshot.edges.push_back(std::move(edge));
                }
            }
            out.nodes.push_back(std::move(snapshot));
        }
        std::ranges::sort(out.nodes, {}, &Graph::Node::observation);
        return out;
    }

    Result result() const {
        int most = 0;
        for (std::size_t i = 0; i < root.keys.size(); ++i) most = std::max(most, visits(root, i));
        Result r{actions[ranked(most, 1).front()], actions, {}, float(root.backed_value)};
        for (double p : Selection::policy(root, s)) r.policy.push_back(float(p));
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
