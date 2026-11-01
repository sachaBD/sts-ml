#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>

namespace stsrl::search {
namespace {
struct Node {
    std::vector<std::uint64_t> keys;  // edges, by public action key
    std::vector<double> logits, prior, value_sum;
    std::vector<int> visits;
    std::vector<std::map<std::uint64_t, std::unique_ptr<Node>>> children;  // per edge, by Chance::child_key
    double value = 0;
    int total = 0;
    int id = 0;  // creation order
};

// How the tree meets hidden information and chance: where a simulation starts, and which children
// a move can lead to. Particles: every simulation plays out in one sampled world, and children are
// told apart by what the player can see, so all worlds share one tree.
class Particles {
public:
    Particles(const sts::BattleContext& observed, int count) : worlds_{combat::particles(observed, count)} {}
    sts::BattleContext start(std::mt19937_64& rng) const {
        return worlds_[std::uniform_int_distribution<std::size_t>{0, worlds_.size() - 1}(rng)];
    }
    static std::uint64_t child_key(const sts::BattleContext& state) { return combat::observation_key(state); }

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

// mctx qtransform_completed_by_mix_value: completed Q (unvisited = mixed value), min-max rescaled, scaled.
std::vector<double> sigma(const Node& n, const Gumbel::Settings& s) {
    double prior_visited = 0, weighted = 0;
    int most = 0;
    for (std::size_t i = 0; i < n.keys.size(); ++i)
        if (n.visits[i]) {
            prior_visited += n.prior[i]; weighted += n.prior[i] * n.value_sum[i] / n.visits[i];
            most = std::max(most, n.visits[i]);
        }
    const double mixed = prior_visited > 0 ? (n.value + n.total * weighted / prior_visited) / (n.total + 1) : n.value;
    std::vector<double> q(n.keys.size());
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = n.visits[i] ? n.value_sum[i] / n.visits[i] : mixed;
    const auto [lo, hi] = std::minmax_element(q.begin(), q.end());
    const double low = *lo, range = std::max(*hi - low, 1e-8);
    for (auto& v : q) v = (s.maxvisit_init + most) * s.value_scale * (v - low) / range;
    return q;
}

std::vector<double> improved_policy(const Node& n, const Gumbel::Settings& s) {
    auto z = sigma(n, s);
    for (std::size_t i = 0; i < z.size(); ++i) z[i] += n.logits[i];
    return softmax(z);
}

// mctx seq_halving.get_sequence_of_considered_visits.
std::vector<int> halving_schedule(int m, int simulations) {
    std::vector<int> out;
    if (m <= 1) { for (int i = 0; i < simulations; ++i) out.push_back(i); return out; }
    const int log2m = int(std::ceil(std::log2(double(m))));
    std::vector<int> visits(m, 0);
    for (int considered = m; int(out.size()) < simulations; considered = std::max(2, considered / 2)) {
        const int extra = std::max(1, int(double(simulations) / (log2m * considered)));
        for (int r = 0; r < extra; ++r) {
            for (int i = 0; i < considered; ++i) out.push_back(visits[i]);
            for (int i = 0; i < considered; ++i) ++visits[i];
        }
    }
    out.resize(simulations);
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
    Settings s;
    std::mt19937_64 rng;
    Chance chance;
    std::vector<Action> actions;
    Node root;
    std::vector<double> gumbel;
    std::vector<int> schedule;
    int simulation = 0, nodes = 1;

    Node* leaf = nullptr;
    sts::BattleContext leaf_state;
    std::vector<Action> leaf_actions;
    std::vector<std::pair<Node*, std::size_t>> path;

    Impl(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
        : s{settings}, rng{seed}, chance{observed, settings.particles}, actions{combat::legal_actions(observed)},
          leaf{&root}, leaf_state{observed}, leaf_actions{actions} {}

    void evaluated(const Evaluation& e) {
        expand(*leaf, e);
        if (leaf == &root) {
            std::extreme_value_distribution<double> g;
            for (std::size_t i = 0; i < actions.size(); ++i) gumbel.push_back(s.gumbel_scale * g(rng));
            schedule = halving_schedule(std::min<int>(s.considered_actions, int(actions.size())), s.simulations);
        } else {
            backup(e.value);
        }
        leaf = nullptr;
        while (!leaf && simulation < s.simulations) simulate();
    }

    // Descends until a finished fight (backed up now) or a new node (left pending for the evaluator).
    void simulate() {
        auto state = chance.start(rng);
        path.clear();
        Node* node = &root;
        for (;;) {
            const auto edge = node == &root ? root_choice(schedule[simulation]) : interior_choice(*node);
            path.emplace_back(node, edge);
            play(state, node->keys[edge]);
            if (state.outcome != sts::Outcome::UNDECIDED) { backup(state.outcome == sts::Outcome::PLAYER_VICTORY); return; }
            auto& child = node->children[edge][Chance::child_key(state)];
            if (!child) {
                child = std::make_unique<Node>();
                child->id = nodes++;
                leaf = child.get();
                leaf_state = state;
                leaf_actions = combat::legal_actions(state);
                return;
            }
            node = child.get();
        }
    }

    void backup(double value) {
        for (auto [n, e] : path) { ++n->visits[e]; n->value_sum[e] += value; ++n->total; }
        ++simulation;
    }

    void expand(Node& n, const Evaluation& e) {
        n.value = e.value;
        n.logits.assign(e.logits.begin(), e.logits.end());
        n.prior = softmax(n.logits);
        for (const auto a : leaf_actions) n.keys.push_back(combat::action_key(leaf_state, a));
        n.value_sum.assign(n.keys.size(), 0); n.visits.assign(n.keys.size(), 0); n.children.resize(n.keys.size());
    }

    // Root: argmax of gumbel + logits + sigma among actions with exactly `considered` visits.
    std::size_t root_choice(int considered) const {
        const auto q = sigma(root, s);
        std::size_t best = 0; double best_score = -INFINITY;
        for (std::size_t i = 0; i < q.size(); ++i)
            if (root.visits[i] == considered && gumbel[i] + root.logits[i] + q[i] > best_score)
                best = i, best_score = gumbel[i] + root.logits[i] + q[i];
        return best;
    }

    // Interior: argmax of improved policy - visits / (1 + total).
    std::size_t interior_choice(const Node& n) const {
        const auto pi = improved_policy(n, s);
        std::size_t best = 0; double best_score = -INFINITY;
        for (std::size_t i = 0; i < pi.size(); ++i) {
            const double score = pi[i] - double(n.visits[i]) / (1 + n.total);
            if (score > best_score) best = i, best_score = score;
        }
        return best;
    }

    Result result() const {
        const int most = *std::max_element(root.visits.begin(), root.visits.end());
        double value_sum = 0;
        for (double v : root.value_sum) value_sum += v;
        Result r{actions[root_choice(most)], actions, {},
                 float(root.total ? value_sum / root.total : root.value)};
        for (double p : improved_policy(root, s)) r.policy.push_back(float(p));
        return r;
    }

    Tree tree() const {
        Tree t{std::vector<Tree::Node>(nodes), gumbel};
        copy(root, t);
        return t;
    }

    void copy(const Node& n, Tree& t) const {
        auto& out = t.nodes[n.id];
        out.value = n.value;
        out.total = n.total;
        const auto q = sigma(n, s);
        const auto pi = improved_policy(n, s);
        for (std::size_t i = 0; i < n.keys.size(); ++i) {
            Tree::Edge e{n.keys[i], n.logits[i], n.prior[i], q[i], pi[i], n.value_sum[i], n.visits[i], {}};
            for (const auto& [observation, child] : n.children[i]) {
                e.children.push_back({observation, child->id});
                copy(*child, t);
            }
            out.edges.push_back(std::move(e));
        }
    }
};

Gumbel::Gumbel(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
    : impl_{std::make_unique<Impl>(observed, settings, seed)} {}
Gumbel::~Gumbel() = default;

std::optional<Query> Gumbel::pending() const {
    if (!impl_->leaf) return std::nullopt;
    return Query{impl_->leaf_state, impl_->leaf_actions};
}

void Gumbel::evaluated(const Evaluation& evaluation) { impl_->evaluated(evaluation); }
Result Gumbel::result() const { return impl_->result(); }
Gumbel::Tree Gumbel::tree() const { return impl_->tree(); }
}  // namespace stsrl::search
