#pragma once
#include "agents/combat/search/filter/belief.hpp"
#include "combat/BattleContext.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace stsrl::search::filter {
// Recursive Kalman-VOI search tree: policy-informed Kalman prior on every edge,
// shared-CRN control variate beta_w from the prior line's Turn+1 world evaluations,
// deduplicated chance outcomes, Bellman max backup, and one unified VOI/cost scan.
class Tree {
public:
    using Key = std::uint64_t;
    using State = sts::BattleContext;

    struct EvalSummary {
        double value = 0, sd = 0;
        std::vector<Key> action_keys;
        std::vector<float> logits;
        std::vector<int> order;
    };

    struct Edge {
        double logit_gap = 0, prior_mean = 0, q = 0, innovation = 0, child_weight = 1.;
        bool chance = false;
        std::vector<int> world_children;   // one child_id per sampled world (0..n-1)
        std::vector<int> unique_children;  // deduplicated child_ids on this edge
    };

    struct Node {
        int id = 0, parent = -1, parent_edge = -1, root_arm = -1, world = -1, depth = 0, turn = 0, evals = 1;
        Key key = 0;
        bool terminal = false, cycle = false, has_root_kalman = false;
        double nn_value = 0, prior_sd = 0, value = 0, path_logit_gap = 0;
        State state;
        std::vector<Key> action_keys;
        std::vector<int> order;
        std::vector<Edge> edges;
    };

    struct Candidate {
        int parent = 0, edge = 0, depth = 0;
        bool world_sample = false;
        double voi = -1., weight = 0., gap = 0., innovation = 0.;
    };

    int add_root(Key key, State state, const EvalSummary& ev) {
        nodes_.push_back({0, -1, -1, -1, -1, 0, state.turn, 1, key, false, false, false, 0., 0., 0., 0.,
                          std::move(state), {}, {}, {}});
        init_node(nodes_[0], ev);
        return 0;
    }

    // Returns {child_id, created}. Reuses an existing child if a chance edge hits the same key again.
    std::pair<int, bool> add_child(int parent_id, int edge_idx, int world, bool chance, Key key, State state,
                                   const EvalSummary* ev) {
        auto& parent = nodes_[parent_id];
        auto& edge = parent.edges[edge_idx];
        if (edge.world_children.empty())
            edge.chance = chance;
        if (chance) {
            for (int existing : edge.unique_children) {
                if (nodes_[existing].key == key) {
                    edge.world_children.push_back(existing);
                    return {existing, false};
                }
            }
        }
        const int id = int(nodes_.size());
        const int depth = parent.depth + 1;
        const int root_arm = parent_id == 0 ? edge_idx : parent.root_arm;
        const int child_world = chance ? world : parent.world;
        const double path_lg = (!chance && parent.world < 0) ? parent.path_logit_gap + edge.logit_gap : 0.;
        const bool terminal = ev == nullptr;
        const bool cycle = !terminal && is_ancestor(parent_id, key);
        nodes_.push_back({id, parent_id, edge_idx, root_arm, child_world, depth, state.turn, 1, key, terminal, cycle,
                          false, 0., 0., 0., path_lg, std::move(state), {}, {}, {}});
        auto& child = nodes_.back();
        if (terminal) {
            child.nn_value = double(child.state.outcome == sts::Outcome::PLAYER_VICTORY);
            child.value = child.nn_value;
        } else {
            init_node(child, *ev);
        }
        auto& pe = nodes_[parent_id].edges[edge_idx];
        pe.world_children.push_back(id);
        pe.unique_children.push_back(id);
        return {id, true};
    }

    // First root-prefix chance edge along the policy-prior line (provides the CRN control variate beta_w).
    std::optional<std::pair<int, int>> reference_chance_edge() const {
        if (nodes_.empty() || nodes_[0].order.empty())
            return std::nullopt;
        int cur = 0;
        for (int hops = 0; hops < 32; ++hops) {
            const auto& n = nodes_[cur];
            if (n.terminal || n.cycle || n.order.empty() || n.world >= 0)
                return std::nullopt;
            const int g = n.order[0];
            const auto& e = n.edges[g];
            if (e.world_children.empty())
                return std::nullopt;
            if (e.chance)
                return std::pair{cur, g};
            cur = e.world_children[0];
        }
        return std::nullopt;
    }

    // Next step needed to evaluate the prior line on shared world `w`, if it hasn't reached `w` yet.
    std::optional<std::pair<int, int>> reference_step_for_world(int w) const {
        if (nodes_.empty() || nodes_[0].order.empty())
            return std::nullopt;
        int cur = 0;
        for (int hops = 0; hops < 32; ++hops) {
            const auto& n = nodes_[cur];
            if (n.terminal || n.cycle || n.order.empty() || n.world >= 0)
                return std::nullopt;
            const int g = n.order[0];
            const auto& e = n.edges[g];
            if (e.world_children.empty())
                return std::pair{cur, g};
            if (e.chance)
                return int(e.world_children.size()) <= w ? std::optional{std::pair{cur, g}} : std::nullopt;
            cur = e.world_children[0];
        }
        return std::nullopt;
    }

    // Follow greedy continuation from `start_id` until Turn+1 boundary (or terminal) to read baseline nn_value.
    double turn1_baseline_nn(int start_id) const {
        const int root_turn = nodes_[0].turn;
        int cur = start_id;
        for (int hops = 0; hops < 32; ++hops) {
            const auto& n = nodes_[cur];
            if (n.turn > root_turn || n.terminal || n.cycle || n.order.empty())
                return n.nn_value;
            const auto& e = n.edges[n.order[0]];
            if (e.world_children.empty())
                return n.nn_value;
            cur = e.world_children[0];
        }
        return nodes_[cur].nn_value;
    }

    void backup(bool fixed_uncertainty) {
        // Control variate beta[w] = prior_line_Turn1_nn(w) - root.nn_value: cancels shared draw luck
        // from world 0 onward without erasing deeper Turn+2/Turn+3 improvements on the prior line itself.
        const auto ref = reference_chance_edge();
        std::vector<double> beta;
        if (ref) {
            const auto& re = nodes_[ref->first].edges[ref->second];
            beta.reserve(re.world_children.size());
            for (int cid : re.world_children)
                beta.push_back(turn1_baseline_nn(cid) - nodes_[0].nn_value);
        }
        const int root_turn = nodes_[0].turn;
        std::vector<double> ys;
        for (int id = int(nodes_.size()) - 1; id >= 0; --id) {
            auto& n = nodes_[id];
            if (n.terminal || n.cycle || n.edges.empty()) {
                n.value = n.nn_value;
                n.has_root_kalman = false;
                n.evals = 1;
                continue;
            }
            const double pv = n.prior_sd * n.prior_sd;
            const double innov0 = n.prior_sd / std::numbers::sqrt2;
            double best = -std::numeric_limits<double>::infinity();
            bool best_has_kalman = false;
            int total_evals = 1;
            double anchor = n.nn_value;
            // Process policy order so greedy (order[0]) updates first and anchors sibling priors to the
            // same Turn+1 scale, eliminating mid-turn nn_value inflation from unexpanded siblings.
            for (const int ei : n.order) {
                auto& e = n.edges[ei];
                const bool greedy_cont = n.id > 0 && ei == n.order[0];
                const bool is_ref_edge = ref && ref->first == id && ref->second == ei;
                e.prior_mean = anchor + kPolicyTemp * e.logit_gap;
                bool edge_has_kalman = false;
                if (e.world_children.empty()) {
                    e.q = e.prior_mean;
                    e.innovation = innov0;
                    e.child_weight = 1.;
                } else if (!e.chance) {
                    const auto& c = nodes_[e.world_children[0]];
                    total_evals += c.evals;
                    edge_has_kalman = c.has_root_kalman;
                    if (c.terminal || greedy_cont || c.has_root_kalman) {
                        e.q = c.value;
                        e.child_weight = 1.;
                    } else {
                        const double w = double(c.evals) / (1. + c.evals);
                        e.q = (1. - w) * e.prior_mean + w * c.value;
                        e.child_weight = w;
                    }
                    e.innovation = 0.;
                } else {
                    ys.clear();
                    const bool shared_root_worlds = n.world < 0;
                    for (int w = 0; w < int(e.world_children.size()); ++w) {
                        const double raw = nodes_[e.world_children[w]].value;
                        const double offset = (shared_root_worlds && w < int(beta.size())) ? beta[w] : 0.;
                        ys.push_back(raw - offset);
                    }
                    for (int cid : e.unique_children)
                        total_evals += nodes_[cid].evals;
                    const double prior = (shared_root_worlds && (n.id == 0 || greedy_cont))
                                             ? nodes_[0].nn_value + kPolicyTemp * (n.path_logit_gap + e.logit_gap)
                                             : e.prior_mean;
                    const double edge_pv = shared_root_worlds ? nodes_[0].prior_sd * nodes_[0].prior_sd : pv;
                    // Unpaired deep chance edges lack CRN cancellation; floor their observation noise by Bernoulli draw variance.
                    const double draw_var = 0.01 * std::max(0., n.nn_value * (1. - n.nn_value));
                    const double obs_pv = shared_root_worlds ? edge_pv : std::max(edge_pv, draw_var);
                    const Belief b = kalman(prior, edge_pv, ys, obs_pv);
                    if (greedy_cont && !shared_root_worlds && n.turn == root_turn) {
                        double sum = 0;
                        for (double y : ys)
                            sum += y;
                        e.q = sum / double(ys.size());
                        e.child_weight = 1. / double(ys.size());
                    } else {
                        e.q = b.mean;
                        e.child_weight = b.weight();
                        edge_has_kalman = shared_root_worlds;
                    }
                    e.innovation = is_ref_edge ? 0. : (fixed_uncertainty ? innov0 : b.innovation);
                }
                if (greedy_cont && !e.world_children.empty())
                    anchor = e.q;
                if (e.q > best) {
                    best = e.q;
                    best_has_kalman = edge_has_kalman;
                }
            }
            n.value = best;
            n.has_root_kalman = best_has_kalman;
            n.evals = total_evals;
        }
    }

    std::optional<Candidate> best_candidate(int max_worlds, bool root_only) const {
        const auto& r = nodes_[0];
        if (r.edges.size() <= 1)
            return std::nullopt;
        const int lead = leader(0);
        int runner_up = -1;
        for (int a : r.order)
            if (a != lead && (runner_up < 0 || r.edges[a].q > r.edges[runner_up].q))
                runner_up = a;
        const double q_lead = r.edges[lead].q;
        const double second = r.edges[runner_up].q;
        // Track each public state's reach weight along the leader's and runner-up's current best lines
        // so shared transpositions (e.g. Strike->Defend vs Defend->Strike) cancel out of root VOI.
        std::unordered_map<Key, double> lead_reach, second_reach;
        if (!root_only) {
            collect_best_reach(0, lead, 1., lead_reach);
            collect_best_reach(0, runner_up, 1., second_reach);
        }
        std::optional<Candidate> best;
        for (int a : r.order) {
            const bool is_lead = a == lead;
            const double gap = is_lead ? std::max(0., q_lead - second) : std::max(0., q_lead - r.edges[a].q);
            const auto& opp_reach = is_lead ? second_reach : lead_reach;
            scan_edge(0, a, is_lead, 1., gap, max_worlds, root_only, opp_reach, best);
        }
        return best;
    }

    int leader(int id = 0) const {
        const auto& n = nodes_[id];
        int best = n.order[0];
        for (int a : n.order)
            if (n.edges[a].q > n.edges[best].q)
                best = a;
        return best;
    }

    std::vector<double> root_q() const {
        std::vector<double> qs;
        qs.reserve(nodes_[0].edges.size());
        for (const auto& e : nodes_[0].edges)
            qs.push_back(e.q);
        return qs;
    }

    bool is_ancestor(int id, Key key) const {
        for (int cur = id; cur >= 0; cur = nodes_[cur].parent)
            if (nodes_[cur].key == key)
                return true;
        return false;
    }

    std::vector<Key> path_keys(int id) const {
        std::vector<Key> path;
        for (int cur = id; cur > 0; cur = nodes_[cur].parent)
            path.push_back(nodes_[nodes_[cur].parent].action_keys[nodes_[cur].parent_edge]);
        std::ranges::reverse(path);
        return path;
    }

    bool empty() const { return nodes_.empty(); }
    const Node& node(int id) const { return nodes_[id]; }
    const std::vector<Node>& nodes() const { return nodes_; }

private:
    std::vector<Node> nodes_;

    static void init_node(Node& n, const EvalSummary& ev) {
        n.nn_value = ev.value;
        n.prior_sd = n.cycle ? 0. : ev.sd;
        n.value = ev.value;
        if (n.cycle)
            return;
        n.action_keys = ev.action_keys;
        n.order = ev.order;
        const float top = ev.logits[ev.order[0]];
        const double innov0 = n.prior_sd / std::numbers::sqrt2;
        n.edges.resize(ev.logits.size());
        for (std::size_t i = 0; i < ev.logits.size(); ++i) {
            const double lg = double(ev.logits[i] - top);
            const double pm = ev.value + kPolicyTemp * lg;
            n.edges[i] = {lg, pm, pm, innov0, 1., false, {}, {}};
        }
    }

    static void consider(std::optional<Candidate>& best, Candidate c) {
        if (c.voi < 0)
            return;
        if (!best || c.voi > best->voi + 1e-12 ||
            (std::abs(c.voi - best->voi) <= 1e-12 && c.depth < best->depth))
            best = c;
    }

    void collect_best_reach(int id, int a, double weight, std::unordered_map<Key, double>& reach) const {
        if (weight <= 1e-9)
            return;
        const auto& e = nodes_[id].edges[a];
        for (int cid : e.unique_children) {
            const int mult = int(std::ranges::count(e.world_children, cid));
            const double cw = weight * e.child_weight * mult;
            const auto& c = nodes_[cid];
            reach[c.key] += cw;
            if (!c.terminal && !c.cycle && !c.edges.empty())
                collect_best_reach(cid, leader(cid), cw, reach);
        }
    }

    void scan_edge(int id, int a, bool down, double weight, double gap, int max_worlds, bool root_only,
                   const std::unordered_map<Key, double>& opp_reach, std::optional<Candidate>& best) const {
        const auto& n = nodes_[id];
        const auto& e = n.edges[a];
        if (e.world_children.empty()) {
            consider(best, {id, a, n.depth + 1, false, positive_part(-gap, weight * e.innovation), weight, gap,
                            e.innovation});
            return;
        }
        if (e.chance && e.innovation > 0 && int(e.world_children.size()) < max_worlds) {
            consider(best, {id, a, n.depth + 1, true, positive_part(-gap, weight * e.innovation), weight, gap,
                            e.innovation});
        }
        if (root_only)
            return;
        for (int cid : e.unique_children) {
            const int mult = int(std::ranges::count(e.world_children, cid));
            scan_node(cid, down, weight * e.child_weight * mult, gap, max_worlds, opp_reach, best);
        }
    }

    void scan_node(int id, bool down, double weight, double gap, int max_worlds,
                   const std::unordered_map<Key, double>& opp_reach, std::optional<Candidate>& best) const {
        const auto& n = nodes_[id];
        if (n.terminal || n.cycle || n.edges.empty())
            return;
        // Cancel reach weight shared with the opposing root arm (transposition invariance).
        double eff_w = weight;
        if (auto it = opp_reach.find(n.key); it != opp_reach.end())
            eff_w = std::max(0., weight - it->second);
        if (eff_w <= 1e-9)
            return;
        if (down) {
            scan_edge(id, leader(id), true, eff_w, gap, max_worlds, false, opp_reach, best);
        } else {
            for (int a : n.order)
                scan_edge(id, a, false, eff_w, gap + eff_w * std::max(0., n.value - n.edges[a].q), max_worlds, false,
                          opp_reach, best);
        }
    }
};
}
