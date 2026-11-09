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
    const double low = *lo, range = s.rescale_values ? std::max(*hi - low, 1e-8) : 1.0;
    for (auto& v : q) v = (s.maxvisit_init + most) * s.value_scale * (v - low) / range;
    return q;
}

std::vector<double> improved_policy(const Node& n, const Gumbel::Settings& s) {
    auto z = sigma(n, s);
    for (std::size_t i = 0; i < z.size(); ++i) z[i] += n.logits[i];
    return softmax(z);
}

// Complete passes only: every competitor receives the same budget before elimination.
struct Round { int actions, visits; };
std::vector<Round> halving_schedule(int m, int simulations) {
    if (simulations <= 0) return {};
    m = std::min(m, std::max(1, simulations / 2));
    if (m <= 1) return {{1, simulations}};
    const int log2m = int(std::ceil(std::log2(double(m))));
    std::vector<Round> out;
    for (int used = 0; used + m <= simulations; m = std::max(2, m / 2)) {
        const int extra = std::max(1, int(double(simulations) / (log2m * m)));
        const int passes = std::min(extra, (simulations - used) / m);
        out.push_back({m, passes});
        used += m * passes;
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
    Settings s;
    std::mt19937_64 rng;
    Chance chance;
    std::vector<Action> actions;
    Node root;
    std::vector<double> gumbel;
    std::vector<Round> schedule;
    std::size_t round = 0;
    int nodes = 1;
    bool root_claimed = false, root_evaluated = false;
    std::vector<std::size_t> competitors;
    std::vector<int> targets;
    std::vector<bool> busy;

    struct Pending {
        Node* leaf;
        sts::BattleContext state;
        std::vector<Action> actions;
        std::vector<std::pair<Node*, std::size_t>> path;
    };
    std::map<std::uint64_t, Pending> pending;

    Impl(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
        : s{settings}, rng{seed}, chance{observed, settings.particles}, actions{combat::legal_actions(observed)} {
        if (s.max_inflight < 1 || s.considered_actions < 1 || s.simulations < 0)
            throw std::invalid_argument{"gumbel: positive inflight/actions and nonnegative simulations required"};
        pending.emplace(0, Pending{&root, observed, actions, {}});
    }

    bool done() const { return root_evaluated && round == schedule.size() && pending.empty(); }

    void begin_round() {
        if (round == schedule.size()) return;
        const auto q = sigma(root, s);
        const int most = *std::max_element(root.visits.begin(), root.visits.end());
        competitors.clear();
        for (std::size_t i = 0; i < actions.size(); ++i)
            if (root.visits[i] == most) competitors.push_back(i);
        std::stable_sort(competitors.begin(), competitors.end(), [&](auto a, auto b) {
            return gumbel[a] + root.logits[a] + q[a] > gumbel[b] + root.logits[b] + q[b];
        });
        competitors.resize(schedule[round].actions);
        targets = root.visits;
        for (auto edge : competitors) targets[edge] += schedule[round].visits;
    }

    void advance_round() {
        if (round < schedule.size() && std::ranges::all_of(competitors, [&](auto e) { return root.visits[e] == targets[e]; })) {
            ++round;
            begin_round();
        }
    }

    std::optional<Query> next_query() {
        if (!root_evaluated) {
            if (root_claimed) return std::nullopt;
            root_claimed = true;
            const auto& p = pending.at(0);
            return Query{p.state, p.actions, 0};
        }
        for (;;) {
            advance_round();
            if (round == schedule.size() || int(pending.size()) >= s.max_inflight) return std::nullopt;
            // Least-visited ready competitor first; a slow evaluation does not stall the others.
            std::optional<std::size_t> edge;
            for (auto e : competitors)
                if (!busy[e] && root.visits[e] < targets[e] && (!edge || root.visits[e] < root.visits[*edge])) edge = e;
            if (!edge) return std::nullopt;
            if (auto query = simulate(*edge)) return query;
        }
    }

    void evaluated(std::uint64_t id, const Evaluation& e) {
        const auto it = pending.find(id);
        if (it == pending.end() || (id == 0 && !root_claimed))
            throw std::logic_error{"gumbel: unknown or duplicate evaluation"};
        const auto& p = it->second;
        if (e.logits.size() != p.actions.size()) throw std::invalid_argument{"gumbel: evaluation action count"};
        expand(*p.leaf, p, e);
        if (id == 0) {
            std::extreme_value_distribution<double> g;
            for (std::size_t i = 0; i < actions.size(); ++i) gumbel.push_back(s.gumbel_scale * g(rng));
            schedule = halving_schedule(std::min<int>(s.considered_actions, int(actions.size())), s.simulations);
            busy.assign(actions.size(), false);
            root_evaluated = true;
            begin_round();
        } else {
            backup(p.path, e.value);
            busy[p.path.front().second] = false;
        }
        pending.erase(it);
        advance_round();
    }

    std::optional<Query> simulate(std::size_t root_edge) {
        auto state = chance.start(rng);
        std::vector<std::pair<Node*, std::size_t>> path;
        Node* node = &root;
        for (;;) {
            const auto edge = node == &root ? root_edge : interior_choice(*node);
            path.emplace_back(node, edge);
            play(state, node->keys[edge]);
            if (state.outcome != sts::Outcome::UNDECIDED) {
                backup(path, state.outcome == sts::Outcome::PLAYER_VICTORY);
                return std::nullopt;
            }
            auto& child = node->children[edge][Chance::child_key(state)];
            if (!child) {
                child = std::make_unique<Node>();
                child->id = nodes++;
                auto [it, inserted] = pending.emplace(child->id, Pending{child.get(), state, combat::legal_actions(state), std::move(path)});
                busy[root_edge] = true;
                const auto& p = it->second;
                return Query{p.state, p.actions, std::uint64_t(child->id)};
            }
            node = child.get();
        }
    }

    static void backup(const std::vector<std::pair<Node*, std::size_t>>& path, double value) {
        for (auto [n, e] : path) { ++n->visits[e]; n->value_sum[e] += value; ++n->total; }
    }

    static void expand(Node& n, const Pending& p, const Evaluation& e) {
        n.value = e.value;
        n.logits.assign(e.logits.begin(), e.logits.end());
        n.prior = softmax(n.logits);
        for (const auto a : p.actions) n.keys.push_back(combat::action_key(p.state, a));
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
        if (!done()) throw std::logic_error{"gumbel: result requested before completion"};
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
        if (n.keys.empty()) return;  // an outstanding leaf has not received its policy yet
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

std::optional<Query> Gumbel::next_query() { return impl_->next_query(); }
void Gumbel::evaluated(std::uint64_t id, const Evaluation& evaluation) { impl_->evaluated(id, evaluation); }
bool Gumbel::done() const { return impl_->done(); }
Result Gumbel::result() const { return impl_->result(); }
Gumbel::Tree Gumbel::tree() const { return impl_->tree(); }
}  // namespace stsrl::search
