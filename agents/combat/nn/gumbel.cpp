#include "agents/combat/nn/gumbel.hpp"
#include "agents/combat/teacher/teacher_leaves.hpp"
#include "sim/search/BattleScumSearcher2.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace stsrl::nn {
namespace {
using sts::search::Action;
using Public = sts::search::PublicBeliefCombatSearch;

struct Node {
    std::vector<std::uint64_t> keys;  // edges, by public action key (legal_actions order at expansion)
    std::vector<double> logits, prior, value_sum;
    std::vector<int> visits;
    std::vector<std::map<std::uint64_t, std::unique_ptr<Node>>> children;  // per edge, by public observation
    double value = 0;                  // network value at expansion
    int total = 0;
};

std::vector<double> softmax(std::vector<double> x) {
    const double top = *std::max_element(x.begin(), x.end());
    double sum = 0;
    for (auto& v : x) sum += v = std::exp(v - top);
    for (auto& v : x) v /= sum;
    return x;
}

// mctx qtransform_completed_by_mix_value: completed Q (unvisited = mixed value), min-max rescaled, scaled.
std::vector<double> sigma(const Node& n, const Settings& s) {
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

std::vector<double> improved_policy(const Node& n, const Settings& s) {
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

// The action with public key `key` in this particle's state (card slots can differ between particles).
void play(sts::BattleContext& state, std::uint64_t key) {
    for (const auto a : legal_actions(state))
        if (Public::publicActionKey(state, a) == key) { a.execute(state); return; }
    throw std::runtime_error{"nn: action missing in a particle"};
}
}  // namespace

struct Search::Impl {
    const Settings& s;
    std::mt19937_64 rng;
    std::vector<sts::BattleContext> particles;
    sts::search::BattleScumSearcher2 rollout;
    std::vector<Action> actions;  // root legal actions
    Node root;
    std::vector<double> gumbel;
    std::vector<int> schedule;
    int simulation = 0;

    // The leaf waiting for the network: its node, state (for the rollout), path to back up, and tokens.
    Node* leaf = nullptr;
    sts::BattleContext leaf_state;
    std::vector<std::pair<Node*, std::size_t>> path;
    stsrl::infra::Tokens tokens;

    Impl(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
        : s{settings}, rng{seed}, particles{teacher::public_particles(observed, settings.particles)},
          rollout{observed}, actions{legal_actions(observed)} {
        rollout.rolloutMode = 2;
        leaf = &root;  // the root is the first evaluation
        leaf_state = observed;
        tokens = encode(observed, actions);
    }

    void evaluated(const stsrl::infra::Prediction& p) {
        expand(*leaf, leaf_state, p);
        if (leaf == &root) {
            std::extreme_value_distribution<double> g;
            for (std::size_t i = 0; i < actions.size(); ++i) gumbel.push_back(s.gumbel_scale * g(rng));
            schedule = halving_schedule(std::min<int>(s.considered, int(actions.size())), s.simulations);
        } else {
            backup(leaf_value(*leaf, leaf_state));
        }
        leaf = nullptr;
        while (!leaf && simulation < s.simulations) simulate();
    }

    // One simulation: descend from the root until a terminal state (backed up now) or a new node (left pending).
    void simulate() {
        auto state = particles[std::uniform_int_distribution<std::size_t>{0, particles.size() - 1}(rng)];
        path.clear();
        Node* node = &root;
        for (;;) {
            const auto edge = node == &root ? root_choice(schedule[simulation]) : interior_choice(*node);
            path.emplace_back(node, edge);
            play(state, node->keys[edge]);
            if (state.outcome != sts::Outcome::UNDECIDED) { backup(state.outcome == sts::Outcome::PLAYER_VICTORY); return; }
            auto& child = node->children[edge][Public::observationKey(state)];
            if (!child) {
                child = std::make_unique<Node>();
                leaf = child.get(); leaf_state = state;
                tokens = encode(state, legal_actions(state));
                return;
            }
            node = child.get();
        }
    }

    void backup(double value) {
        for (auto [n, e] : path) { ++n->visits[e]; n->value_sum[e] += value; ++n->total; }
        ++simulation;
    }

    void expand(Node& n, const sts::BattleContext& state, const stsrl::infra::Prediction& p) {
        n.value = p.value;
        n.logits.assign(p.logits.begin(), p.logits.end());
        n.prior = softmax(n.logits);
        for (const auto a : legal_actions(state)) n.keys.push_back(Public::publicActionKey(state, a));
        n.value_sum.assign(n.keys.size(), 0); n.visits.assign(n.keys.size(), 0); n.children.resize(n.keys.size());
    }

    double leaf_value(const Node& n, sts::BattleContext state) {
        if (s.rollout_mix == 0) return n.value;
        rollout.randGen.seed(rng());
        sts::search::BattleScumSearcher2::Node scratch;
        for (int i = 0; i < 512 && state.outcome == sts::Outcome::UNDECIDED; ++i) rollout.rolloutAction(scratch, state).execute(state);
        return (1 - s.rollout_mix) * n.value + s.rollout_mix * (state.outcome == sts::Outcome::PLAYER_VICTORY);
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
        Result r{actions[root_choice(most)], actions, {}};
        for (double p : improved_policy(root, s)) r.policy.push_back(float(p));
        return r;
    }
};

Search::Search(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed)
    : impl_{std::make_unique<Impl>(observed, settings, seed)} {}
Search::~Search() = default;
const stsrl::infra::Tokens* Search::pending() const { return impl_->leaf ? &impl_->tokens : nullptr; }
void Search::evaluated(const stsrl::infra::Prediction& prediction) { impl_->evaluated(prediction); }
Result Search::result() const { return impl_->result(); }
}  // namespace stsrl::nn
