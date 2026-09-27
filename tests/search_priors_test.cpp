// PublicBeliefCombatSearch policy-prior mode: expand at evaluation, per-node priors, PUCT with
// first-play urgency. (Mode off is checked bit-for-bit against the historical search separately.)
#include "agents/teacher_search.hpp"
#include "scenarios/slime_boss.hpp"

#include "combat/BattleContext.h"
#include "sim/search/PublicBeliefCombatSearch.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using sts::search::PublicBeliefCombatSearch;
using Search = PublicBeliefCombatSearch;
using namespace stsrl;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        std::exit(1);
    }
}

template <class F>
bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

Search make(int particles, double reduction = 0.15, double alpha = 0, double epsilon = 0) {
    auto search = teacher::make_search(scenarios::slime_boss(3).battle(), false, particles);
    search.setObjective(1, 35, 4, 0, 0);
    search.enablePolicyPriors(1.5, reduction, alpha, epsilon, 11);
    return search;
}

using ValueFn = std::function<double(const Search::Request&)>;
using PriorFn = std::function<std::vector<double>(const Search::Request&)>;

std::vector<double> uniform(const Search::Request& r) { return std::vector<double>(r.child->edges.size(), 1.0); }

void run(Search& s, std::int64_t budget, int batch, const ValueFn& value, const PriorFn& priors = uniform) {
    while (s.simulations < budget) {
        for (const auto id : s.requestBatch(batch, budget, 0, 0)) {
            const auto& r = s.pending.at(id);
            if (r.child) s.submit(id, value(r), priors(r));
            else s.submit(id, value(r));
        }
    }
}

double constant(const Search::Request&) { return 0.5; }

std::size_t most_visited(const Search::Node& n) {
    return std::max_element(n.edges.begin(), n.edges.end(),
                            [](const auto& a, const auto& b) { return a.visits < b.visits; }) - n.edges.begin();
}

void test_api_guards() {
    auto legacy = teacher::make_search(scenarios::slime_boss(3).battle(), false, 4);
    check(throws([&] { legacy.enablePolicyPriors(1.5, 0.15); }), "policy priors require objectiveMode 1");
    legacy.setObjective(1, 35, 4, 0, 0);
    legacy.setRootPrior(std::vector<double>(legacy.root().edges.size(), 1.0), 1.0);
    check(throws([&] { legacy.enablePolicyPriors(1.5, 0.15); }), "policy priors refuse a root prior");

    auto s = make(4);
    check(throws([&] { s.setRootPrior(std::vector<double>(s.root().edges.size(), 1.0), 1.0); }),
          "root prior refused in policy-prior mode");
    check(throws([&] { s.search(10); }), "search() refused in policy-prior mode");
    check(throws([&] { s.requestBatch(4, 10, 1, 1); }), "rollout bounds refused in policy-prior mode");
    check(throws([&] { s.setObjective(0, 35, 4, 0, 0); }), "objective 0 refused in policy-prior mode");
    check(s.objectiveMaxHp() == s.particles.front().player.maxHp, "objectiveMaxHp is the root max HP");

    // The root's own evaluation comes first, alone.
    const auto first = s.requestBatch(16, 100, 0, 0);
    check(first.size() == 1, "first batch is the root alone");
    const auto& r = s.pending.at(first[0]);
    check(r.path.empty() && r.child == &s.root(), "first request is the root");
    check(throws([&] { s.submit(first[0], 0.5, std::vector<double>(s.root().edges.size() + 1, 1.0)); }),
          "wrong prior count refused");
    check(throws([&] { s.submit(first[0], 0.5, std::vector<double>(s.root().edges.size(), 0.0)); }),
          "all-zero priors refused");
    std::vector<double> p(s.root().edges.size(), 0.0);
    p[0] = 3.0;
    s.submit(first[0], 0.4, p);
    check(s.root().evaluated && s.root().value == 0.4 && s.root().edges[0].prior == 1.0,
          "root priors normalized, value stored");
}

// (b) A prior near 1 dominates the early visits.
void test_prior_steers() {
    auto s = make(4);
    std::size_t favoured = 0;
    run(s, 1, 1, constant, [&](const Search::Request& r) {
        auto p = std::vector<double>(r.child->edges.size(), 0.001);
        favoured = p.size() - 1;
        p[favoured] = 1.0;
        return p;
    });
    check(s.root().edges.size() > 2, "enough root edges");
    run(s, 101, 8, constant);
    check(most_visited(s.root()) == favoured, "prior ~1 edge is the most visited");
    check(s.root().edges[favoured].visits >= 60, "prior ~1 edge takes most early visits");
}

// (b') Values override priors: with an optimistic first-play value, zero-prior edges are tried and
// the one with the best leaves takes the visits from the favoured one.
void test_value_overrides_prior() {
    auto s = make(4, -0.5);
    run(s, 1, 1, constant, [](const Search::Request& r) {
        auto p = std::vector<double>(r.child->edges.size(), 0.0);
        p[0] = 1.0;
        return p;
    });
    const std::size_t good = s.root().edges.size() - 1;
    const auto* root = &s.root();
    run(s, 400, 1, [&](const Search::Request& r) {
        return !r.path.empty() && r.path[0].first == root && r.path[0].second == good ? 0.9 : 0.1;
    });
    check(most_visited(s.root()) == good, "zero-prior edge with high Q wins the visits");
}

// (d) First-play urgency, no forced first visit: a pessimistic FPU keeps revisiting the first
// edge tried; an optimistic one tries every edge before revisiting any.
void test_fpu() {
    auto pessimistic = make(4, 10.0);
    run(pessimistic, 50, 1, constant);
    const auto& edges = pessimistic.root().edges;
    check(std::count_if(edges.begin(), edges.end(), [](const auto& e) { return e.visits > 0; }) == 1,
          "pessimistic FPU: unvisited edges are not forced");

    auto optimistic = make(4, -10.0);
    const auto n = static_cast<std::int64_t>(optimistic.root().edges.size());
    run(optimistic, 1 + n, 1, constant);
    for (const auto& e : optimistic.root().edges) check(e.visits == 1, "optimistic FPU: every edge once");
}

// (c) + (e) Every node is created by exactly one request and ends up evaluated, including new chance
// outcomes of already-visited edges; batch siblings descend into still-pending nodes instead of
// evaluating them twice.
void test_expansion_invariants() {
    auto s = make(8);
    bool descended_into_pending = false;
    const auto root = s.requestBatch(64, 3000, 0, 0);
    s.submit(root.at(0), 0.5, uniform(s.pending.at(root.at(0))));
    while (s.simulations < 3000) {
        const auto ids = s.requestBatch(64, 3000, 0, 0);
        std::set<const Search::Node*> children;
        for (const auto id : ids) {
            const auto* child = s.pending.at(id).child;
            if (child) check(children.insert(child).second, "one request per node");
        }
        for (const auto id : ids)
            for (const auto& [node, edge] : s.pending.at(id).path)
                descended_into_pending |= children.count(node) > 0;
        for (const auto id : ids) {
            const auto& r = s.pending.at(id);
            if (!r.child) { s.submit(id, 0.5); continue; }
            auto p = uniform(r);
            p[0] = 2.0;
            s.submit(id, 0.3 + 0.01 * (r.path.size() % 7), p);
        }
    }
    check(descended_into_pending, "batch siblings descend into pending nodes");
    bool chance = false;
    for (const auto& [key, node] : s.nodes) {
        check(node->evaluated, "every node evaluated");
        const auto sum = std::accumulate(node->edges.begin(), node->edges.end(), 0.0,
                                         [](double a, const auto& e) { return a + e.prior; });
        check(std::abs(sum - 1) < 1e-9, "priors normalized");
        const auto visited = std::count_if(node->edges.begin(), node->edges.end(), [](const auto& e) { return e.visits > 0; });
        chance |= static_cast<std::int64_t>(node->children.size()) > visited;
    }
    check(chance, "some visited edge reached several public children");

    // Rebase keeps the retained root's priors; an unseen new root is requested first.
    const auto before = s.particles.front();
    const auto action = s.selectedAction();
    auto after = before;
    action.execute(after);
    check(after.outcome == sts::Outcome::UNDECIDED, "fight continues");
    s.rebase({after}, s.actionKey(before, action), 5);
    check(s.root().evaluated, "retained root keeps its evaluation");
    check(s.requestBatch(8, 100, 0, 0).size() > 1, "retained root is not re-requested");
}

void test_dirichlet() {
    auto s = make(4, 0.15, 0.3, 0.25);
    run(s, 1, 1, constant);
    const auto& edges = s.root().edges;
    const double u = 1.0 / edges.size();
    double sum = 0;
    bool moved = false;
    for (const auto& e : edges) { sum += e.prior; moved |= std::abs(e.prior - u) > 1e-6; }
    check(moved && std::abs(sum - 1) < 1e-9, "root noise mixed in and normalized");
}

}  // namespace

int main() {
    test_api_guards();
    test_prior_steers();
    test_value_overrides_prior();
    test_fpu();
    test_expansion_invariants();
    test_dirichlet();
    std::cout << "search_priors_test passed\n";
}
